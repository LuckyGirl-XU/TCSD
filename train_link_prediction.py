from models.TCSD import TCSD
import logging
import time
import sys
import os
import numpy as np
import warnings
import json
import torch
import torch.nn as nn
os.environ['WANDB__SERVICE_WAIT'] = '300'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
from models.modules import MergeLayer, MulMergeLayer, BPRLoss, InfoNCELoss
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction
from evaluate_models_utils import evaluate_model_link_prediction_multi_negs
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args

def train_epoch(model, args, logger, epoch, train_idx_data_loader, train_neighbor_sampler, train_neg_edge_sampler, train_data, optimizer, loss_func, full_neighbor_sampler, val_data, val_idx_data_loader, val_neg_edge_sampler, full_data):
    model.train()
    (train_losses, train_metrics) = ([], [])
    for (batch_idx, train_data_indices) in enumerate(train_idx_data_loader):
        train_data_indices = train_data_indices.numpy()
        (batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids) = (train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices])
        if args.collision_check:
            batch_neg_dst_node_ids = train_neg_edge_sampler.sample_with_time_collision_check(num_negs=1, batch_src_node_ids=batch_src_node_ids, batch_node_interact_times=batch_node_interact_times, neighbor_sampler=train_neighbor_sampler).flatten()
            batch_neg_src_node_ids = batch_src_node_ids
        else:
            (batch_neg_src_node_ids, batch_neg_dst_node_ids) = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
        (src_neighb_seq, src_edge_ids, src_neighb_interact_times) = train_neighbor_sampler.get_historical_neighbors_left(node_ids=batch_src_node_ids, node_interact_times=batch_node_interact_times, num_neighbors=args.num_neighbors)
        src_hop_ids = None
        neighbor_num = (src_neighb_seq != 0).sum(axis=1)
        if neighbor_num.sum() == 0:
            continue
        pos_item = torch.from_numpy(batch_dst_node_ids).unsqueeze(-1)
        if args.num_train_negs > 1:
            if args.collision_check:
                multi_neg_ids = train_neg_edge_sampler.sample_with_time_collision_check(num_negs=args.num_train_negs, batch_src_node_ids=batch_src_node_ids, batch_node_interact_times=batch_node_interact_times, neighbor_sampler=train_neighbor_sampler)
                neg_item = torch.from_numpy(multi_neg_ids)
            else:
                neg_items_list = []
                for _ in range(args.num_train_negs):
                    (_, neg_dst) = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                    neg_items_list.append(neg_dst)
                neg_item = torch.from_numpy(np.stack(neg_items_list, axis=1))
        else:
            neg_item = torch.from_numpy(batch_neg_dst_node_ids).unsqueeze(-1)
        test_dst = torch.cat([pos_item, neg_item], dim=-1)
        (dst_last_neighbor, _, dst_last_update_time) = train_neighbor_sampler.get_historical_neighbors_left(node_ids=test_dst.flatten(), node_interact_times=np.broadcast_to(batch_node_interact_times[:, np.newaxis], (len(batch_node_interact_times), test_dst.shape[1])).flatten(), num_neighbors=1)
        dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
        dst_last_update_time[dst_last_neighbor.reshape(len(test_dst), -1) == 0] = -100000
        dst_last_update_time = torch.from_numpy(dst_last_update_time)
        dst_num_neighbors = min(args.num_neighbors, 50)
        test_dst_np = test_dst.numpy()
        dst_neighb_list = []
        dst_times_list = []
        for d_col in range(test_dst_np.shape[1]):
            (dn_seq, _, dn_times) = train_neighbor_sampler.get_historical_neighbors_left(node_ids=test_dst_np[:, d_col], node_interact_times=batch_node_interact_times, num_neighbors=dst_num_neighbors)
            dst_neighb_list.append(dn_seq)
            dst_times_list.append(dn_times)
        dst_neighb_seq = torch.from_numpy(np.stack(dst_neighb_list, axis=1))
        dst_neighb_times = torch.from_numpy(np.stack(dst_times_list, axis=1))
        (loss, predicts, labels) = model.calculate_loss(src_neighb_seq=torch.from_numpy(src_neighb_seq), src_neighb_seq_len=torch.from_numpy(neighbor_num), src_neighb_interact_times=torch.from_numpy(src_neighb_interact_times), cur_pred_times=torch.from_numpy(batch_node_interact_times), test_dst=test_dst, dst_last_update_times=dst_last_update_time, edge_ids=torch.from_numpy(src_edge_ids), dst_neighb_seq=dst_neighb_seq, dst_neighb_interact_times=dst_neighb_times, **{'src_hop_ids': src_hop_ids} if src_hop_ids is not None else {})
        if predicts is not None and labels is not None:
            train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    logger.info(f"Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]['lr']}, train loss: {np.mean(train_losses):.8f}")
    if scheduler is not None:
        scheduler.step()
    if len(train_metrics) > 0:
        for metric_name in train_metrics[0].keys():
            logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.8f}')
    if args.use_mrr_val:
        val_metrics = evaluate_model_link_prediction_multi_negs(model_name=args.model_name, model=model, neighbor_sampler=full_neighbor_sampler, evaluate_idx_data_loader=val_idx_data_loader, evaluate_neg_edge_sampler=val_neg_edge_sampler, evaluate_data=val_data, loss_func=loss_func, device=args.device, num_neighbors=args.num_neighbors, time_gap=args.time_gap, mode='val', loss_type=args.loss, full_data=full_data, dataset_name=args.dataset_name, collision_check=args.collision_check, momentum=args.momentum)
    else:
        (val_losses, val_metrics) = evaluate_model_link_prediction(model_name=args.model_name, model=model, neighbor_sampler=full_neighbor_sampler, evaluate_idx_data_loader=val_idx_data_loader, evaluate_neg_edge_sampler=val_neg_edge_sampler, evaluate_data=val_data, loss_func=loss_func, device=args.device, num_neighbors=args.num_neighbors, time_gap=args.time_gap, mode='val', loss_type=args.loss, full_data=full_data, collision_check=args.collision_check, dataset_name=args.dataset_name, momentum=args.momentum)
    for metric_name in val_metrics.keys():
        logger.info(f'validate {metric_name}, {val_metrics[metric_name]:.8f}')
    for handler in logger.handlers:
        handler.flush()
    return val_metrics

def get_model(args, train_data, node_raw_features, edge_raw_features, train_neighbor_sampler, full_data, logger):
    args.item_size = max(args.item_size, getattr(args, 'node_size', args.item_size))
    dynamic_backbone = TCSD(args.num_layers, args.num_heads, args.embedding_size, args.hidden_dropout, args.attn_dropout_prob, args.hidden_act, args.layer_norm_eps, args.initializer_range, args.item_size, max_seq_length=args.num_neighbors, device=args.device, loss_type=args.loss, use_pos=args.use_pos, input_cat_time_intervals=args.input_cat_time_intervals, output_cat_time_intervals=args.output_cat_time_intervals, output_cat_repeat_times=args.output_cat_repeat_times, num_output_layer=args.num_output_layer, emb_dropout_prob=args.emb_dropout_prob, skip_connection=args.skip_connection, latent_dim=args.latent_dim, kl_weight=args.kl_weight, num_latent_modes=args.num_latent_modes, use_structural_invariance=args.use_structural_invariance, si_weight=args.si_weight, edge_feat_dim=edge_raw_features.shape[1] if args.use_edge_feat and edge_raw_features is not None and (edge_raw_features.shape[1] > 0) else 0)
    dynamic_backbone._craft_num_hops = getattr(args, 'num_hops', 1)
    dynamic_backbone._craft_hop_nn = args.num_neighbors // max(getattr(args, 'num_hops', 1), 1)
    if args.merge in ['cat']:
        link_predictor = MergeLayer(input_dim1=args.output_dim, input_dim2=args.output_dim, hidden_dim=args.output_dim, output_dim=1)
    elif args.merge in ['mul']:
        link_predictor = MulMergeLayer(scale=args.scale)
    dynamic_backbone.set_min_idx(src_min_idx=args.src_min_idx, dst_min_idx=args.dst_min_idx)
    if hasattr(dynamic_backbone, 'set_edge_features'):
        dynamic_backbone.set_edge_features(edge_raw_features)
    model = dynamic_backbone
    logger.info(f'model -> {model}')
    logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, {get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')
    return model

def get_loss_fn(args):
    if args.loss in ['BPR']:
        loss_func = BPRLoss()
    elif args.loss in ['InfoNCE']:
        loss_func = InfoNCELoss(temperature=0.1)
    elif args.loss in ['BCE']:
        loss_func = nn.BCELoss()
    else:
        raise NotImplementedError(f'Loss function {args.loss} not implemented!')
    return loss_func
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = get_link_prediction_args(is_evaluation=False)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    postfix = ''
    if args.use_edge_feat:
        postfix += '_e'
    if args.use_node_feat:
        postfix += '_n'
    if hasattr(args, 'run_name') and args.run_name:
        postfix += f'_{args.run_name}'
    args.save_model_name = f'{args.model_name}_seed{args.seed}_batchsize{args.batch_size}_num_neighbors{args.num_neighbors}_dropout{args.dropout}_sample_neighbor_strategy{args.sample_neighbor_strategy}_numlayers{args.num_layers}{postfix}'
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    log_dir = f'{current_dir}/logs/{args.model_name}/{args.dataset_name}'
    os.makedirs(log_dir, exist_ok=True)
    log_file = f'{log_dir}/{str(time.time())}.log'
    print('log in: ', log_file)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    (node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, val_test_data) = get_link_prediction_data(dataset_name=args.dataset_name, dataset_path=args.dataset_path, use_edge_feat=args.use_edge_feat, use_node_feat=args.use_node_feat, logger=logger)
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=0)
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=1)
    if args.is_bipartite or args.dataset_name in ['GoogleLocal', 'ML-20M', 'Taobao', 'Yelp', 'mooc', 'lastfm', 'reddit', 'wikipedia']:
        args.user_size = full_data.src_node_ids.max() - full_data.src_node_ids.min() + 1
        args.item_size = full_data.dst_node_ids.max() - full_data.dst_node_ids.min() + 1
        args.node_size = args.user_size + args.item_size
        args.dst_min_idx = full_data.dst_node_ids.min()
        args.src_min_idx = full_data.src_node_ids.min()
    else:
        args.user_size = full_data.max_node_id
        args.item_size = full_data.max_node_id
        args.node_size = args.user_size
        args.dst_min_idx = 1
        args.src_min_idx = 1
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids, seed=0)
    val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=1)
    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    if args.dataset_name in ['wikipedia', 'reddit', 'mooc', 'lastfm', 'uci', 'Flights']:
        args.collision_check = True
    train_indices_list = list(range(len(train_data.src_node_ids)))
    train_idx_data_loader = get_idx_data_loader(indices_list=train_indices_list, batch_size=args.batch_size)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size)
    test_indices_list = list(range(len(test_data.src_node_ids)))
    test_idx_data_loader = get_idx_data_loader(indices_list=test_indices_list, batch_size=args.batch_size)
    multi_negs_test_idx_data_loader = get_idx_data_loader(indices_list=test_indices_list, batch_size=args.multi_negs_batch_size)
    (val_metric_all_runs, test_metric_all_runs) = ([], [])
    for run in range(args.num_runs):
        set_random_seed(seed=args.seed + run)
        run_start_time = time.time()
        logger.info(f'********** Run {run + 1} starts. **********')
        logger.info(f'configuration is {args}')
        logger.info(f'{sys.argv}')
        random_projections = None
        model = get_model(args, train_data, node_raw_features, edge_raw_features, train_neighbor_sampler, full_data, logger)
        loss_func = get_loss_fn(args)
        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = None
        if hasattr(args, 'use_lr_scheduler') and args.use_lr_scheduler:
            warmup_epochs = min(5, args.num_epochs // 5)

            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                else:
                    import math
                    progress = (epoch - warmup_epochs) / max(1, args.num_epochs - warmup_epochs)
                    return 0.5 * (1 + math.cos(math.pi * progress))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        args.save_model_name = f'{args.model_name}_seed{args.seed + run}_batchsize{args.batch_size}_num_neighbors{args.num_neighbors}_dropout{args.dropout}_numlayers{args.num_layers}{postfix}'
        save_model_folder = f'{args.save_model_path}/{args.dataset_name}/{args.model_name}/{args.save_model_name}/'
        if not os.path.exists(save_model_folder):
            os.makedirs(save_model_folder, exist_ok=True)
        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder, save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)
        if args.load_pretrained:
            if not os.path.isfile(early_stopping.save_model_path):
                raise FileNotFoundError(f'Requested checkpoint does not exist: {early_stopping.save_model_path}')
            early_stopping.load_checkpoint(model, map_location='cpu')
        model = convert_to_gpu(model, device=args.device)
        for epoch in range(args.num_epochs):
            val_metrics = train_epoch(model, args, logger, epoch, train_idx_data_loader, train_neighbor_sampler, train_neg_edge_sampler, train_data, optimizer, loss_func, full_neighbor_sampler, val_data, val_idx_data_loader, val_neg_edge_sampler, full_data)
            if 'mrr' in val_metrics:
                val_metric_indicator = [('mrr', val_metrics['mrr'], True)]
            elif 'average_precision' in val_metrics:
                val_metric_indicator = [('average_precision', val_metrics['average_precision'], True)]
            else:
                raise ValueError(f'No valid metric found in val_metrics: {val_metrics}')
            early_stop = early_stopping.step(val_metric_indicator, model)
            if early_stop:
                break
        early_stopping.load_checkpoint(model, map_location='cpu')
        model = convert_to_gpu(model, device=args.device)
        logger.info(f'get final performance on dataset {args.dataset_name}...')
        test_metrics = {}
        (test_losses, test_metrics) = evaluate_model_link_prediction(model_name=args.model_name, model=model, neighbor_sampler=full_neighbor_sampler, evaluate_idx_data_loader=test_idx_data_loader, evaluate_neg_edge_sampler=test_neg_edge_sampler, evaluate_data=test_data, loss_func=loss_func, device=args.device, num_neighbors=args.num_neighbors, time_gap=args.time_gap, mode='test', loss_type=args.loss, full_data=full_data, collision_check=args.collision_check, dataset_name=args.dataset_name, random_projections=random_projections, momentum=args.momentum)
        early_stopping.load_checkpoint(model, map_location='cpu')
        model = convert_to_gpu(model, device=args.device)
        for metric_name in test_metrics.keys():
            logger.info(f'test {metric_name}, {test_metrics[metric_name]:.8f}')
        test_metrics_multi_negs = evaluate_model_link_prediction_multi_negs(model_name=args.model_name, model=model, neighbor_sampler=full_neighbor_sampler, evaluate_idx_data_loader=multi_negs_test_idx_data_loader, evaluate_neg_edge_sampler=test_neg_edge_sampler, evaluate_data=test_data, loss_func=loss_func, device=args.device, num_neighbors=args.num_neighbors, time_gap=args.time_gap, loss_type=args.loss, full_data=full_data, dataset_name=args.dataset_name, collision_check=args.collision_check, random_projections=random_projections, momentum=args.momentum)
        test_metrics.update(test_metrics_multi_negs)
        for metric_name in test_metrics.keys():
            logger.info(f'test {metric_name}, {test_metrics[metric_name]:.8f}')
        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')
        test_metric_all_runs.append(test_metrics)
        result_json = {'test metrics': {metric_name: f'{test_metrics[metric_name]:.8f}' for metric_name in test_metrics}}
        result_json = json.dumps(result_json, indent=4)
        save_result_folder = f'./saved_results/{args.dataset_name}/{args.model_name}'
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f'{args.save_model_name}.json')
        with open(save_result_path, 'w') as file:
            file.write(result_json)
    logger.info(f'metrics over {args.num_runs} runs:')
    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.8f} ± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.8f}')
    print(log_file)
    sys.exit()
