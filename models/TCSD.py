import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.modules import DestinationHistoryPooler, GatedLinearAttention, SinusoidalTimeEncoder, init_weights
from utils.utils import NeighborSampler


class StructuralDynamicsEncoder(nn.Module):
    """
    Structural dynamics modeling (Eq. 6) and the structural gate (Eq. 7).
    """

    def __init__(self, hidden_size, num_indicators=4, dropout=0.1):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(num_indicators, num_indicators, kernel_size=3, padding=1, groups=num_indicators)
        self.pointwise_conv = nn.Sequential(
            nn.Conv1d(num_indicators, num_indicators, kernel_size=1),
            nn.GELU(),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(num_indicators * 2, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Dropout(dropout),
        )
        # start from an open gate
        with torch.no_grad():
            self.gate_mlp[-2].bias.fill_(2.0)

    def forward(self, structural_states, pad_mask):
        """
        :param structural_states: Tensor, shape (batch_size, num_candidates, seq_len, num_indicators)
        :param pad_mask: Tensor, shape (batch_size, seq_len)
        :return: Tensor, shape (batch_size, num_candidates, seq_len, hidden_size), the gate of each neighbor
        """
        batch_size, num_candidates, seq_len, num_indicators = structural_states.shape
        states = structural_states.reshape(-1, seq_len, num_indicators).transpose(1, 2)
        evolved_states = self.pointwise_conv(self.depthwise_conv(states)).transpose(1, 2).reshape(batch_size, num_candidates, seq_len, num_indicators)
        evolved_states = evolved_states.masked_fill(pad_mask.unsqueeze(1).unsqueeze(-1), 0.0)
        return torch.sigmoid(self.gate_mlp(torch.cat([structural_states, evolved_states], dim=-1)))


class TCSD(nn.Module):
    """
    Target-Conditioned Structural Dynamics for link prediction in dynamic graphs.
    For each candidate destination, TCSD builds hybrid structural indicators over the source history (Eq. 1-4),
    models their dynamics along the history (Eq. 6), and uses them to gate and pool the neighbors (Eq. 7-8).
    """

    def __init__(self, neighbor_sampler: NeighborSampler, n_layers: int, n_heads: int, hidden_size: int, hidden_dropout_prob: float, layer_norm_eps: float, initializer_range: float, n_nodes: int, max_seq_length: int, dst_min_idx: int, device: str = 'cpu', emb_dropout_prob: float = 0.1, edge_raw_features: np.ndarray = None, neighbor_drop_rate: float = 0.15):
        """
        TCSD model.
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :param n_layers: int, number of encoder layers
        :param n_heads: int, number of attention heads
        :param hidden_size: int, hidden dimension
        :param hidden_dropout_prob: float, dropout rate of the hidden layers
        :param layer_norm_eps: float, epsilon of the layer normalization
        :param initializer_range: float, standard deviation of the weight initialization
        :param n_nodes: int, size of the node embedding table
        :param max_seq_length: int, length of the source history
        :param dst_min_idx: int, smallest destination id, node ids are shifted so that it maps to 1
        :param device: str, device
        :param emb_dropout_prob: float, dropout rate of the destination representation
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim), None to ignore edge features
        :param neighbor_drop_rate: float, share of the source neighbors dropped during training
        """
        super().__init__()
        self.neighbor_sampler = neighbor_sampler
        self.hidden_size = hidden_size
        self.dst_min_idx = dst_min_idx
        self.neighbor_drop_rate = neighbor_drop_rate
        self.device = device
        edge_feat_dim = 0 if edge_raw_features is None else edge_raw_features.shape[1]
        self.edge_feat_dim = edge_feat_dim

        self.node_embedding = nn.Embedding(n_nodes + 1, hidden_size, padding_idx=0)
        self.time_encoder = SinusoidalTimeEncoder(hidden_size)
        self.rank_embedding = nn.Embedding(max_seq_length + 1, hidden_size, padding_idx=0)

        proj_input_dim = hidden_size * 3
        if edge_feat_dim > 0:
            proj_input_dim += edge_feat_dim
            self.edge_feat_proj = nn.Linear(edge_feat_dim, edge_feat_dim)
            self.register_buffer("edge_features_table", torch.from_numpy(edge_raw_features).float())
        self.neighbor_time_proj = nn.Sequential(
            nn.LayerNorm(proj_input_dim),
            nn.Linear(proj_input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout_prob),
        )

        self.sequence_layers = nn.ModuleList([GatedLinearAttention(hidden_size, n_heads, hidden_dropout_prob) for _ in range(n_layers)])
        self.structural_dynamics = StructuralDynamicsEncoder(hidden_size, num_indicators=4, dropout=hidden_dropout_prob)
        # log of the learnable time scale of the exact co-neighbor indicator (Eq. 2)
        self.co_neighbor_log_scale = nn.Parameter(torch.tensor(5.0))

        self.attentive_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.emb_dropout = nn.Dropout(emb_dropout_prob)
        self.dst_time_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(hidden_dropout_prob),
        )
        self.dst_pooler = DestinationHistoryPooler(hidden_size, dropout=hidden_dropout_prob)

        # the structural dynamics encoder deliberately keeps its own initialization, including the bias of its gate
        for name, module in self.named_children():
            if name != 'structural_dynamics':
                module.apply(lambda layer: init_weights(layer, initializer_range))

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        compute source and destination node temporal embeddings
        The source representation depends on the destination, so all the candidates of a query go through one pass.
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, num_candidates)
        :param node_interact_times: ndarray, shape (batch_size, )
        :param num_neighbors: int, length of the source history, the destination history uses at most 50 neighbors
        :return: two Tensors, with shape (batch_size, num_candidates, hidden_size)
        """
        sampler, num_candidates = self.neighbor_sampler, dst_node_ids.shape[1]
        src_neighb_seq, edge_ids, src_neighb_interact_times = sampler.get_historical_neighbors_left(node_ids=src_node_ids, node_interact_times=node_interact_times, num_neighbors=num_neighbors)

        # last update of each candidate
        dst_last_neighbor, _, dst_last_update_times = sampler.get_historical_neighbors_left(node_ids=dst_node_ids.flatten(), node_interact_times=np.repeat(node_interact_times, num_candidates), num_neighbors=1)
        dst_last_update_times = dst_last_update_times.reshape(len(dst_node_ids), -1)
        dst_last_update_times[dst_last_neighbor.reshape(len(dst_node_ids), -1) == 0] = -100000

        # history of each candidate
        dst_history = [sampler.get_historical_neighbors_left(node_ids=dst_node_ids[:, candidate], node_interact_times=node_interact_times, num_neighbors=min(num_neighbors, 50)) for candidate in range(num_candidates)]
        dst_neighb_seq = torch.from_numpy(np.stack([history[0] for history in dst_history], axis=1)).to(self.device) - self.dst_min_idx + 1
        dst_neighb_interact_times = torch.from_numpy(np.stack([history[2] for history in dst_history], axis=1)).to(self.device)

        src_neighb_seq = torch.from_numpy(src_neighb_seq).to(self.device) - self.dst_min_idx + 1
        src_neighb_seq[src_neighb_seq < 0] = 0
        dst_neighb_seq[dst_neighb_seq < 0] = 0
        return self.forward(src_neighb_seq, torch.from_numpy(src_neighb_interact_times).to(self.device), torch.from_numpy(node_interact_times).to(self.device),
                            torch.from_numpy(dst_node_ids).to(self.device) - self.dst_min_idx + 1, torch.from_numpy(dst_last_update_times).to(self.device),
                            dst_neighb_seq, dst_neighb_interact_times, edge_ids=torch.from_numpy(edge_ids).to(self.device))

    @staticmethod
    def _per_query_time_scale(time_diffs, pad_mask):
        """
        Lower median of each query's valid history, with a minimum of 1.
        Do not pool across the batch: a later query can contain interactions that were not yet observable at an
        earlier query's prediction time. Padding is excluded, and queries without history use a scale of 1.
        """
        ordered = time_diffs.masked_fill(pad_mask, float("inf")).sort(dim=1).values
        counts = (~pad_mask).sum(dim=1, keepdim=True)
        median_indices = (counts - 1).clamp(min=0) // 2
        medians = ordered.gather(1, median_indices)
        medians = torch.where(counts > 0, medians, torch.ones_like(medians))
        return medians.clamp(min=1.0)

    @staticmethod
    def compute_temporal_ranks(time_diffs, pad_mask):
        """
        rank of each neighbor by recency, from 1 for the most recent one, 0 for padding
        """
        order = time_diffs.masked_fill(pad_mask, 1e12).argsort(dim=1)
        return (order.argsort(dim=1) + 1).masked_fill(pad_mask, 0)

    def exact_repeat(self, src_neighb_seq, test_dst, time_diffs, pad_mask):
        """
        Eq. 1, time-aware exact repeat indicator
        """
        repeat_match = (src_neighb_seq.unsqueeze(1) == test_dst.unsqueeze(2)) & (src_neighb_seq.unsqueeze(1) > 0)
        time_scale = self._per_query_time_scale(time_diffs, pad_mask)
        time_weight = torch.exp(-time_diffs / time_scale).masked_fill(pad_mask, 0.0)
        return repeat_match.float() * time_weight.unsqueeze(1)

    def exact_co_neighbor(self, src_neighb_seq, src_times, dst_neighb_seq, dst_times, pad_mask):
        """
        Eq. 2, time-aware exact co-neighbor indicator
        """
        src_ids = src_neighb_seq.unsqueeze(1).unsqueeze(3)
        dst_ids = dst_neighb_seq.unsqueeze(2)
        match = (src_ids == dst_ids) & (src_ids > 0) & (dst_ids > 0)
        time_gap = (src_times.unsqueeze(1).unsqueeze(3) - dst_times.unsqueeze(2)).abs().float().clamp(max=1e10)
        best_time_gap, _ = time_gap.masked_fill(~match, 1e12).min(dim=-1)
        scale = torch.exp(self.co_neighbor_log_scale).clamp(min=1.0)
        indicator = torch.exp(-best_time_gap / scale).masked_fill(~match.any(dim=-1), 0.0)
        return indicator.masked_fill(pad_mask.unsqueeze(1), 0.0)

    @staticmethod
    def relaxed_repeat(src_neighb_norm, dst_norm, pad_mask):
        """
        Eq. 3, relaxed repeat indicator
        """
        return torch.einsum("bsd,bnd->bns", src_neighb_norm, dst_norm).masked_fill(pad_mask.unsqueeze(1), 0.0)

    def relaxed_co_neighbor(self, src_neighb_norm, dst_neighb_seq, pad_mask):
        """
        Eq. 3, relaxed co-neighbor indicator
        Padded pairs take the value -2, the clamp brings them back to 0 so that no padded position leaks into the structural state.
        """
        dst_neighb_norm = F.normalize(self.node_embedding(dst_neighb_seq), dim=-1)
        similarity = torch.einsum("bsh,bdlh->bdsl", src_neighb_norm, dst_neighb_norm)
        similarity = similarity.masked_fill(pad_mask.unsqueeze(1).unsqueeze(-1), -2.0)
        similarity = similarity.masked_fill((dst_neighb_seq == 0).unsqueeze(2), -2.0)
        indicator, _ = similarity.max(dim=-1)
        return indicator.clamp(min=0.0).masked_fill(pad_mask.unsqueeze(1), 0.0)

    def encode_source_history(self, neighb_emb, time_diffs, pad_mask, edge_ids):
        """
        encode the source history, neighbors are randomly dropped during training
        :return: the encoded history, the padding mask after dropping, the mean-pooled source context
        """
        batch_size, seq_len = pad_mask.shape
        rank_feat = self.rank_embedding(self.compute_temporal_ranks(time_diffs, pad_mask))
        neighbor_feats = [neighb_emb, self.time_encoder(time_diffs), rank_feat]
        if self.edge_feat_dim > 0:
            neighbor_feats.append(self.edge_feat_proj(self.edge_features_table[edge_ids.long()]))
        input_emb = self.neighbor_time_proj(torch.cat(neighbor_feats, dim=-1))

        if self.training and self.neighbor_drop_rate > 0:
            drop_mask = torch.rand(batch_size, seq_len, device=self.device) < self.neighbor_drop_rate
            drop_mask = drop_mask & ~pad_mask
            # keep at least one neighbor of each source with a history
            for idx in (pad_mask | drop_mask).all(dim=1).nonzero(as_tuple=True)[0]:
                first_valid = (~pad_mask[idx]).nonzero(as_tuple=True)[0]
                if len(first_valid) > 0:
                    drop_mask[idx, first_valid[0]] = False
            pad_mask = pad_mask | drop_mask

        seq_out = input_emb.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        for seq_layer in self.sequence_layers:
            seq_out = seq_layer(seq_out, pad_mask)
            seq_out = torch.nan_to_num(seq_out, nan=0.0)
            seq_out = seq_out.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        active = (~pad_mask).unsqueeze(-1).float()
        src_ctx = (seq_out * active).sum(dim=1) / active.sum(dim=1).clamp(min=1.0)
        return seq_out, pad_mask, src_ctx

    def encode_destination(self, test_dst, src_ctx, cur_times, dst_last_update_times, dst_neighb_seq, dst_neighb_interact_times):
        """
        destination representation from its embedding, its last update and its pooled history
        :return: the raw embeddings and the representation of the candidates
        """
        batch_size = test_dst.shape[0]
        dst_emb = self.node_embedding(test_dst).view(batch_size, -1, self.hidden_size)
        dt_dst = cur_times.view(-1, 1) - dst_last_update_times
        dt_dst[dst_last_update_times < -1] = 0
        dst_time_ctx = self.dst_time_proj(self.time_encoder(dt_dst.float().clamp(min=0, max=1e10)))

        dt_dst_neighb = (cur_times.view(-1, 1, 1) - dst_neighb_interact_times).float().clamp(min=0, max=1e10)
        dst_pool = self.dst_pooler(src_ctx, self.node_embedding(dst_neighb_seq), self.time_encoder(dt_dst_neighb), dst_neighb_seq == 0)
        return dst_emb, self.emb_dropout(self.LayerNorm(dst_emb + dst_time_ctx + dst_pool))

    def forward(self, src_neighb_seq, neighbors_interact_times, cur_times, test_dst, dst_last_update_times, dst_neighb_seq, dst_neighb_interact_times, edge_ids=None):
        """
        :param src_neighb_seq: Tensor, shape (batch_size, seq_len), the source history, 0 for padding
        :param neighbors_interact_times: Tensor, shape (batch_size, seq_len)
        :param cur_times: Tensor, shape (batch_size, ), prediction times
        :param test_dst: Tensor, shape (batch_size, num_candidates)
        :param dst_last_update_times: Tensor, shape (batch_size, num_candidates)
        :param dst_neighb_seq: Tensor, shape (batch_size, num_candidates, dst_seq_len), the history of the candidates
        :param dst_neighb_interact_times: Tensor, shape (batch_size, num_candidates, dst_seq_len)
        :param edge_ids: Tensor, shape (batch_size, seq_len), only used with edge features
        :return: two Tensors, with shape (batch_size, num_candidates, hidden_size), the source and destination embeddings
        """
        batch_size, num_candidates = test_dst.shape
        neighb_emb = self.node_embedding(src_neighb_seq)
        time_diffs = (cur_times.view(-1, 1) - neighbors_interact_times).float().clamp(min=0, max=1e10)
        seq_out, pad_mask, src_ctx = self.encode_source_history(neighb_emb, time_diffs, src_neighb_seq == 0, edge_ids)
        dst_emb, dst_repr = self.encode_destination(test_dst, src_ctx, cur_times, dst_last_update_times, dst_neighb_seq, dst_neighb_interact_times)

        # hybrid structural states (Eq. 4), the order of the channels is part of the learned weights
        src_neighb_norm = F.normalize(neighb_emb, dim=-1)
        structural_states = torch.stack([
            self.relaxed_repeat(src_neighb_norm, F.normalize(dst_emb, dim=-1), pad_mask),
            self.exact_co_neighbor(src_neighb_seq, neighbors_interact_times, dst_neighb_seq, dst_neighb_interact_times, pad_mask),
            self.exact_repeat(src_neighb_seq, test_dst, time_diffs, pad_mask),
            self.relaxed_co_neighbor(src_neighb_norm, dst_neighb_seq, pad_mask),
        ], dim=-1)

        # structural dynamics-guided aggregation (Eq. 6-8)
        gated_neighbors = seq_out.unsqueeze(1).expand(-1, num_candidates, -1, -1) * self.structural_dynamics(structural_states, pad_mask)
        pool_mask = pad_mask.unsqueeze(1).expand(-1, num_candidates, -1)
        pool_scores = self.attentive_pool(gated_neighbors).squeeze(-1).masked_fill(pool_mask, -1e9)
        pool_weights = torch.softmax(pool_scores, dim=-1).masked_fill(pool_mask, 0.0)
        src_pooled = torch.bmm(
            pool_weights.view(batch_size * num_candidates, 1, -1),
            gated_neighbors.view(batch_size * num_candidates, -1, self.hidden_size),
        ).view(batch_size, num_candidates, self.hidden_size)

        return src_pooled, dst_repr
