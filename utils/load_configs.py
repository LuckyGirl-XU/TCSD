"""TCSD dataset presets, with explicit command-line overrides."""
import argparse
import json
from pathlib import Path
import sys
import torch

def get_link_prediction_args(is_evaluation=False, argv=None):
    parser = argparse.ArgumentParser('Interface for the link prediction task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', required=True)
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--multi_negs_batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--model_name', type=str, default='TCSD', choices=['TCSD'], help='model name')
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--output_dim', type=int, default=172, help='dimension of the output embedding')
    parser.add_argument('--embedding_size', type=int, default=172, help='dimension of the output embedding')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--num_train_negs', type=int, default=1, help='Number of negative samples per positive during training for TCSD')
    parser.add_argument('--use_lr_scheduler', action=argparse.BooleanOptionalAction, default=False, help='Use cosine annealing LR scheduler with warmup')
    parser.add_argument('--save_model_path', type=str, default='./saved_models/')
    parser.add_argument('--dataset_path', type=str, default='./data/')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-06, type=float, help='the hyperparameter that controls the sampling preference with time interval, a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--time_gap', type=int, default=20000, help='time gap for neighbors to compute node features')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--scale', type=float, default=12, help='scale the scores')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=15, help='patience for early stopping')
    parser.add_argument('--num_runs', type=int, default=1, help='number of runs')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--is_bipartite', type=bool, default=False, help='the dataset is bipartite')
    parser.add_argument('--use_edge_feat', action=argparse.BooleanOptionalAction, default=False, help='whether to use edge feat')
    parser.add_argument('--load_pretrained', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--use_node_feat', action=argparse.BooleanOptionalAction, default=False, help='whether to use node feat')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--merge', type=str, default='cat', help='model for merge two embeddings')
    parser.add_argument('--hidden_dropout', type=float, default=0.5, help='dropout rate for hidden layers')
    parser.add_argument('--hidden_act', type=str, default='gelu', help='activation function for hidden layers')
    parser.add_argument('--loss', type=str, default='BCE', help='loss function type')
    parser.add_argument('--attn_dropout_prob', type=float, default=0.5, help='dropout rate for attention layers')
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12, help='epsilon for layer normalization')
    parser.add_argument('--initializer_range', type=float, default=0.02, help='stddev for initializing weight')
    parser.add_argument('--use_pos', action=argparse.BooleanOptionalAction, default=False, help='whether to use position embedding')
    parser.add_argument('--output_cat_time_intervals', action=argparse.BooleanOptionalAction, default=False, help='whether to concatenate elapsed time encoding in TCSD')
    parser.add_argument('--output_cat_repeat_times', action=argparse.BooleanOptionalAction, default=False, help='whether to concatenate repeat time encoding in TCSD')
    parser.add_argument('--input_cat_time_intervals', action=argparse.BooleanOptionalAction, default=False, help='whether to use time encoding instead of positional encoding in TCSD')
    parser.add_argument('--num_output_layer', type=int, default=1, help='number of output layers in TCSD')
    parser.add_argument('--collision_check', action=argparse.BooleanOptionalAction, default=False, help='whether to check collision in train')
    parser.add_argument('--emb_dropout_prob', type=float, default=0.1, help='dropout rate for embedding')
    parser.add_argument('--use_mrr_val', action=argparse.BooleanOptionalAction, default=False, help='whether to use mrr for validation')
    parser.add_argument('--skip_connection', action=argparse.BooleanOptionalAction, default=False, help='whether to use skip connection in TCSD')
    parser.add_argument('--run_name', type=str, default='', help='Run name for logging')
    parser.add_argument('--momentum', type=float, default=0.1, help='momentum for memory update EMA')
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--kl_weight', type=float, default=0.01)
    parser.add_argument('--num_latent_modes', type=int, default=4)
    parser.add_argument('--use_structural_invariance', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--si_weight', type=float, default=0.1)
    parser.add_argument('--print_config', action='store_true', help='print resolved settings and exit without loading data or training')
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--dataset_name')
    selected, _ = probe.parse_known_args(argv)
    if selected.dataset_name is None:
        parser.parse_args(argv)  # --help, or argparse's required-dataset error.
    plan = json.loads((Path(__file__).resolve().parents[1] / 'configs' / 'datasets.json').read_text())
    aliases = {rec['dataset_name']: name for name, rec in plan['datasets'].items()}
    name = aliases.get(selected.dataset_name, selected.dataset_name)
    if name not in plan['datasets']:
        parser.error('No reproduction preset for dataset: ' + name + '. Supported: ' + ', '.join(plan['datasets']))
    rec = plan['datasets'][name]
    defaults = dict(plan['common'])
    defaults.update(rec['config'])
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.dataset_name = rec['dataset_name']
    args.device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'
    if args.num_runs < 1 or args.num_epochs < 0:
        parser.error('num_runs must be positive and num_epochs must be non-negative')
    if args.num_epochs == 0 and not args.load_pretrained:
        parser.error('Evaluation with num_epochs=0 requires --load_pretrained')
    if args.print_config:
        print(json.dumps(vars(args), indent=2))
        parser.exit()
    return args
