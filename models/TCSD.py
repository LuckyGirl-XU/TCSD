import math


import torch


from torch import nn


import torch.nn.functional as F


def get_activation(act_type="PReLU", pinit=0.15):
    act_type = act_type.lower()
    if act_type == "relu":
        return nn.ReLU()
    if act_type == "gelu":
        return nn.GELU()
    if act_type == "prelu":
        return nn.PReLU(init=pinit)
    raise ValueError(f"Unsupported activation: {act_type}")


class MLP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        act_type="PReLU",
        dropout=0.1,
        use_act=True,
        skip_connection=True,
        pinit=0.15,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.lins = nn.ModuleList()
        self.linear_b4_skip = None
        if num_layers > 1:
            cur_dim = input_dim
            for _ in range(num_layers - 1):
                self.lins.append(nn.Linear(cur_dim, hidden_dim))
                cur_dim = hidden_dim
            self.lins.append(nn.Linear(cur_dim, output_dim))
            self.dropout = nn.Dropout(dropout)
            self.act = get_activation(act_type, pinit)
            self.use_act = use_act
            self.skip_connection = skip_connection
            if skip_connection and input_dim != output_dim:
                self.linear_b4_skip = nn.Linear(input_dim, output_dim)
        else:
            self.lins.append(nn.Linear(input_dim, output_dim))

    def forward(self, inputs):
        x = inputs
        if self.num_layers > 1:
            for i in range(len(self.lins) - 1):
                x = self.lins[i](x)
                if self.use_act:
                    x = self.act(x)
                x = self.dropout(x)
            x = self.lins[-1](x)
            if self.skip_connection:
                residual = self.linear_b4_skip(inputs) if self.linear_b4_skip is not None else inputs
                x = x + residual
        else:
            x = self.lins[0](x)
        return x


class GatedLinearAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.g_proj = nn.Linear(hidden_size, n_heads)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x, pad_mask):
        batch_size, seq_len, hidden_dim = x.shape
        residual = x
        x = self.norm(x)

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        gate = torch.sigmoid(self.g_proj(x))

        q_act = F.elu(q.permute(0, 2, 1, 3)) + 1
        k_act = F.elu(k.permute(0, 2, 1, 3)) + 1
        v_perm = v.permute(0, 2, 1, 3)
        gate_perm = gate.permute(0, 2, 1).unsqueeze(-1)

        k_gated = k_act * gate_perm
        v_gated = v_perm * gate_perm

        mask = (~pad_mask).float().unsqueeze(1).unsqueeze(-1)
        k_gated = k_gated * mask
        v_gated = v_gated * mask

        kv = torch.matmul(k_gated.transpose(-2, -1), v_gated)
        k_sum = k_gated.sum(dim=2, keepdim=True)

        out = torch.matmul(q_act, kv)
        normalizer = torch.matmul(q_act, k_sum.transpose(-2, -1)).clamp(min=1e-6)
        out = out / normalizer

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_dim)
        out = self.out_proj(out)
        out = self.dropout(out)
        out = out.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        x = residual + out
        x = x + self.ffn(x)
        return x


class TemporalResonanceEncoder(nn.Module):
    def __init__(self, hidden_size, num_features=5, dropout=0.1):
        super().__init__()
        feat_dim = num_features
        self.tre_depthwise = nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim)
        self.tre_pointwise = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, kernel_size=1),
            nn.GELU(),
        )
        self.gate_proj = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            self.gate_proj[-2].bias.fill_(2.0)
        self.score_proj = nn.Sequential(
            nn.Linear(feat_dim * 2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, srv, pad_mask=None):
        batch_size, num_dst, seq_len, feat_dim = srv.shape
        srv_flat = srv.reshape(batch_size * num_dst, seq_len, feat_dim).transpose(1, 2)
        if pad_mask is not None:
            mask_exp = pad_mask.unsqueeze(1).expand(-1, num_dst, -1).reshape(batch_size * num_dst, seq_len).unsqueeze(1)
            srv_flat = srv_flat.masked_fill(mask_exp, 0.0)
        evolved = self.tre_depthwise(srv_flat)
        evolved = self.tre_pointwise(evolved)
        evolved = evolved.transpose(1, 2).reshape(batch_size, num_dst, seq_len, feat_dim)
        if pad_mask is not None:
            evolved = evolved.masked_fill(pad_mask.unsqueeze(1).unsqueeze(-1), 0.0)
        combined = torch.cat([srv, evolved], dim=-1)
        gate = torch.sigmoid(self.gate_proj(combined))
        score = self.score_proj(combined).squeeze(-1)
        return gate, score


class SrcConditionedDstPooler(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.neighb_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.src_query_proj = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** -0.5
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, src_ctx, dst_neighb_emb, dst_neighb_time_enc, dst_neighb_pad_mask):
        combined = torch.cat([dst_neighb_emb, dst_neighb_time_enc], dim=-1)
        kv = self.neighb_proj(combined)
        q = self.src_query_proj(src_ctx)
        scores = torch.einsum("bh,bdlh->bdl", q, kv) * self.scale
        scores = scores.masked_fill(dst_neighb_pad_mask, -1e9)
        all_pad = dst_neighb_pad_mask.all(dim=-1, keepdim=True)
        weights = torch.softmax(scores, dim=-1).masked_fill(dst_neighb_pad_mask, 0.0)
        dst_pool = torch.einsum("bdl,bdlh->bdh", weights, kv).masked_fill(all_pad, 0.0)
        return self.out_proj(dst_pool)


class SRVForecaster(nn.Module):
    def __init__(self, num_features=5, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.causal_conv = nn.Conv1d(num_features, hidden_dim, kernel_size=3, padding=2)
        self.predictor = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, num_features, kernel_size=1),
        )
        self.forecast_score = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, srv, pad_mask=None):
        batch_size, num_dst, seq_len, feat_dim = srv.shape
        x = srv.reshape(batch_size * num_dst, seq_len, feat_dim).transpose(1, 2)
        if pad_mask is not None:
            mask = pad_mask.unsqueeze(1).expand(-1, num_dst, -1).reshape(batch_size * num_dst, seq_len).unsqueeze(1)
            x = x.masked_fill(mask, 0.0)
        hidden = self.causal_conv(x)[:, :, :seq_len]
        pred = self.predictor(hidden)
        srv_pred = pred.transpose(1, 2).reshape(batch_size, num_dst, seq_len, feat_dim)
        if pad_mask is not None:
            srv_pred = srv_pred.masked_fill(pad_mask.unsqueeze(1).unsqueeze(-1), 0.0)
        delta = srv_pred - srv
        forecast_score = self.forecast_score(delta).squeeze(-1)
        return srv_pred, forecast_score


class _TCSDBase(torch.nn.Module):
    def __init__(
        self,
        n_layers,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
        initializer_range,
        n_nodes,
        max_seq_length,
        device,
        loss_type,
        use_pos=True,
        input_cat_time_intervals=False,
        output_cat_time_intervals=False,
        output_cat_repeat_times=False,
        num_output_layer=1,
        emb_dropout_prob=0.1,
        skip_connection=False,
        latent_dim=32,
        kl_weight=0.05,
        num_latent_modes=4,
        use_structural_invariance=False,
        si_weight=0.1,
        edge_feat_dim=0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_nodes = n_nodes
        self.max_seq_length = max_seq_length
        self.initializer_range = initializer_range
        self.edge_feat_dim = edge_feat_dim
        self.device = device
        self.loss_type = "BCE"
        self.neighbor_drop_rate = 0.15
        self.stf_loss_weight = 0.1
        self.aux_loss = 0.0

        self.node_embedding = nn.Embedding(n_nodes + 1, hidden_size, padding_idx=0)
        time_freqs = torch.linspace(0, 9, hidden_size // 2)
        self.register_buffer("time_freq", (1.0 / 10 ** time_freqs).view(1, 1, -1))
        self.rank_embedding = nn.Embedding(max_seq_length + 1, hidden_size, padding_idx=0)

        if edge_feat_dim > 0:
            proj_input_dim = hidden_size * 3 + edge_feat_dim
            self.edge_feat_proj = nn.Linear(edge_feat_dim, edge_feat_dim)
        else:
            proj_input_dim = hidden_size * 3
        self.neighbor_time_proj = nn.Sequential(
            nn.LayerNorm(proj_input_dim),
            nn.Linear(proj_input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout_prob),
        )

        self.sequence_layers = nn.ModuleList(
            [GatedLinearAttention(hidden_size, n_heads, hidden_dropout_prob) for _ in range(n_layers)]
        )
        self.resonance_encoder = TemporalResonanceEncoder(hidden_size, num_features=5, dropout=hidden_dropout_prob)
        self.srv_forecaster = SRVForecaster(num_features=5, hidden_dim=32, dropout=hidden_dropout_prob)
        self.bridge_log_scale = nn.Parameter(torch.tensor(5.0))

        self.pool_attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.emb_dropout = nn.Dropout(emb_dropout_prob)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.dst_time_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(hidden_dropout_prob),
        )
        self.dst_pooler = SrcConditionedDstPooler(hidden_size, dropout=hidden_dropout_prob)
        self.score_proj = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout_prob),
        )
        self.output_layer = MLP(
            num_layers=num_output_layer,
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=1,
            dropout=hidden_dropout_prob,
            use_act=True,
            skip_connection=skip_connection,
        )

        if loss_type == "BCE":
            self.loss_fct = nn.BCELoss()
        elif loss_type == "BPR":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss type for TCSD: {loss_type}")

        self.apply(self._init_weights)

    def set_min_idx(self, src_min_idx, dst_min_idx):
        self.src_min_idx = src_min_idx
        self.dst_min_idx = dst_min_idx

    def set_edge_features(self, edge_raw_features):
        if self.edge_feat_dim > 0 and edge_raw_features is not None and edge_raw_features.shape[1] > 0:
            self.register_buffer("edge_features_table", torch.from_numpy(edge_raw_features).float())
        else:
            self.edge_features_table = None

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def encode_time(self, dt):
        t = dt.unsqueeze(-1).float()
        angles = t * self.time_freq
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    @staticmethod
    def _per_query_time_scale(time_diffs, pad_mask):
        """Lower median of each query's valid history, with a minimum of 1.

        Do not pool across the batch: a later query can contain interactions
        that were not yet observable at an earlier query's prediction time.
        Padding is excluded, and queries without history use a scale of 1.
        """
        if time_diffs.shape[1] == 0:
            return time_diffs.new_ones((time_diffs.shape[0], 1))
        ordered = time_diffs.masked_fill(pad_mask, float("inf")).sort(dim=1).values
        counts = (~pad_mask).sum(dim=1, keepdim=True)
        median_indices = (counts - 1).clamp(min=0) // 2
        medians = ordered.gather(1, median_indices)
        medians = torch.where(counts > 0, medians, torch.ones_like(medians))
        return medians.clamp(min=1.0)

    def compute_temporal_ranks(self, time_diffs, pad_mask):
        batch_size, seq_len = time_diffs.shape
        td = time_diffs.clone().float()
        td[pad_mask] = 1e12
        _, sorted_indices = td.sort(dim=1)
        ranks = torch.zeros_like(sorted_indices)
        batch_idx = torch.arange(batch_size, device=time_diffs.device).unsqueeze(1).expand_as(sorted_indices)
        ranks[batch_idx, sorted_indices] = torch.arange(1, seq_len + 1, device=time_diffs.device).unsqueeze(0).expand(batch_size, -1)
        ranks[pad_mask] = 0
        return ranks

    def _compute_bridge_score(self, src_neighb_seq, src_times, dst_neighb_seq, dst_times, src_pad_mask):
        src_ids = src_neighb_seq.unsqueeze(1).unsqueeze(3)
        dst_ids = dst_neighb_seq.unsqueeze(2)
        match = (src_ids == dst_ids) & (src_ids > 0) & (dst_ids > 0)

        src_t = src_times.unsqueeze(1).unsqueeze(3)
        dst_t = dst_times.unsqueeze(2)
        time_diff = (src_t - dst_t).abs().float().clamp(max=1e10)

        time_diff_masked = time_diff.masked_fill(~match, 1e12)
        best_time_diff, _ = time_diff_masked.min(dim=-1)
        has_match = match.any(dim=-1)

        scale = torch.exp(self.bridge_log_scale).clamp(min=1.0)
        bridge_score = torch.exp(-best_time_diff / scale)
        bridge_score = bridge_score.masked_fill(~has_match, 0.0)
        bridge_score = bridge_score.masked_fill(src_pad_mask.unsqueeze(1), 0.0)
        return bridge_score

    def forward(
        self,
        src_neighb_seq,
        src_neighb_seq_len,
        neighbors_interact_times,
        cur_times,
        test_dst=None,
        dst_last_update_times=None,
        edge_ids=None,
        dst_neighb_seq=None,
        dst_neighb_interact_times=None,
    ):
        batch_size = src_neighb_seq.shape[0]
        num_dst = test_dst.shape[1]
        src_edge_summary = None
        src_neighb_seq_len = src_neighb_seq_len.clone()
        src_neighb_seq_len[src_neighb_seq_len == 0] = 1

        neighb_emb = self.node_embedding(src_neighb_seq)
        time_diffs = (cur_times.view(-1, 1) - neighbors_interact_times).to(self.device).float().clamp(min=0, max=1e10)
        time_feat = self.encode_time(time_diffs)
        pad_mask = src_neighb_seq == 0

        ranks = self.compute_temporal_ranks(time_diffs, pad_mask)
        rank_feat = self.rank_embedding(ranks.clamp(max=self.max_seq_length))

        if self.edge_feat_dim > 0 and hasattr(self, "edge_features_table") and self.edge_features_table is not None and edge_ids is not None:
            edge_feats = self.edge_features_table[edge_ids.long()]
            edge_feat_transformed = self.edge_feat_proj(edge_feats)
            src_edge_summary = edge_feat_transformed.abs().mean(dim=-1)
            input_emb = self.neighbor_time_proj(torch.cat([neighb_emb, time_feat, rank_feat, edge_feat_transformed], dim=-1))
        else:
            input_emb = self.neighbor_time_proj(torch.cat([neighb_emb, time_feat, rank_feat], dim=-1))

        if self.training and self.neighbor_drop_rate > 0:
            drop_mask = torch.rand(batch_size, src_neighb_seq.shape[1], device=self.device) < self.neighbor_drop_rate
            drop_mask = drop_mask & ~pad_mask
            all_dropped = (pad_mask | drop_mask).all(dim=1)
            if all_dropped.any():
                for idx in all_dropped.nonzero(as_tuple=True)[0]:
                    first_valid = (~pad_mask[idx]).nonzero(as_tuple=True)[0]
                    if len(first_valid) > 0:
                        drop_mask[idx, first_valid[0]] = False
            pad_mask = pad_mask | drop_mask

        input_emb = input_emb.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        seq_out = input_emb
        for seq_layer in self.sequence_layers:
            seq_out = seq_layer(seq_out, pad_mask)
            seq_out = torch.nan_to_num(seq_out, nan=0.0)
            seq_out = seq_out.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        active_float = (~pad_mask).unsqueeze(-1).float()
        src_ctx = (seq_out * active_float).sum(dim=1) / active_float.sum(dim=1).clamp(min=1.0)

        test_dst_emb = self.node_embedding(test_dst)
        dst_raw = test_dst_emb.view(batch_size, -1, self.hidden_size)
        dt_dst_raw = cur_times.view(-1, 1) - dst_last_update_times
        dt_dst_raw[dst_last_update_times < -1] = 0
        dt_dst_raw = dt_dst_raw.to(self.device).float().clamp(min=0, max=1e10)
        dst_time_enc = self.encode_time(dt_dst_raw)
        dst_time_ctx = self.dst_time_proj(dst_time_enc)

        if dst_neighb_seq is not None:
            dst_neighb_emb_pool = self.node_embedding(dst_neighb_seq)
            dt_dst_neighb = cur_times.view(-1, 1, 1) - dst_neighb_interact_times
            dt_dst_neighb = dt_dst_neighb.to(self.device).float().clamp(min=0, max=1e10)
            dst_neighb_time_enc = self.encode_time(dt_dst_neighb)
            dst_neighb_pad = dst_neighb_seq == 0
            dst_pool = self.dst_pooler(src_ctx, dst_neighb_emb_pool, dst_neighb_time_enc, dst_neighb_pad)
            dst_repr = self.LayerNorm(dst_raw + dst_time_ctx + dst_pool)
        else:
            dst_repr = self.LayerNorm(dst_raw + dst_time_ctx)
        dst_repr = self.emb_dropout(dst_repr)

        seq_expanded = seq_out.unsqueeze(1).expand(-1, num_dst, -1, -1)
        if dst_neighb_seq is not None:
            src_norm = F.normalize(neighb_emb, dim=-1)
            dst_norm = F.normalize(dst_raw, dim=-1)
            w = torch.einsum("bsd,bnd->bns", src_norm, dst_norm).masked_fill(pad_mask.unsqueeze(1), 0.0)

            d = self._compute_bridge_score(
                src_neighb_seq,
                neighbors_interact_times.to(self.device),
                dst_neighb_seq,
                dst_neighb_interact_times,
                pad_mask,
            )

            src_ids = src_neighb_seq.unsqueeze(1)
            dst_ids = test_dst.unsqueeze(2)
            repeat_match = (src_ids == dst_ids) & (src_ids > 0)
            time_median = self._per_query_time_scale(time_diffs, pad_mask)
            time_weight = torch.exp(-time_diffs.clamp(min=0) / time_median).masked_fill(pad_mask, 0.0)
            r = repeat_match.float() * time_weight.unsqueeze(1)

            dst_neighb_emb = self.node_embedding(dst_neighb_seq)
            dst_nb_norm = F.normalize(dst_neighb_emb, dim=-1)
            cross_sim = torch.einsum("bsh,bdlh->bdsl", src_norm, dst_nb_norm)
            dst_pad = dst_neighb_seq == 0
            cross_sim = cross_sim.masked_fill(pad_mask.unsqueeze(1).unsqueeze(-1), -2.0)
            cross_sim = cross_sim.masked_fill(dst_pad.unsqueeze(2), -2.0)
            s, _ = cross_sim.max(dim=-1)
            s = s.clamp(min=0.0).masked_fill(pad_mask.unsqueeze(1), 0.0)

            src_times_exp = neighbors_interact_times.to(self.device).float()
            dst_times_raw = dst_neighb_interact_times.to(self.device).float()
            t_diff = (src_times_exp.unsqueeze(1).unsqueeze(-1) - dst_times_raw.unsqueeze(2)).abs()
            t_diff = t_diff.masked_fill(dst_pad.unsqueeze(2), 1e10)
            min_diff, _ = t_diff.min(dim=-1)
            tca = torch.zeros_like(w).masked_fill(pad_mask.unsqueeze(1), 0.0)

            srv = torch.stack([w, d, r, s, tca], dim=-1)
            gate, pos_score = self.resonance_encoder(srv, pad_mask)
            srv_pred, forecast_sc = self.srv_forecaster(srv, pad_mask)

            if self.training and srv.shape[2] > 1:
                pred_shift = srv_pred[:, :, :-1, :]
                target_shift = srv[:, :, 1:, :]
                valid_mask = ~pad_mask[:, 1:].unsqueeze(1).unsqueeze(-1)
                mse = ((pred_shift - target_shift) ** 2) * valid_mask.float()
                n_valid = valid_mask.float().sum().clamp(min=1.0)
                self.aux_loss = mse.sum() / (n_valid * srv.shape[1] * 5)
            else:
                self.aux_loss = 0.0

            enriched = seq_expanded * gate
            pos_score = pos_score.masked_fill(pad_mask.unsqueeze(1), 0.0)
            forecast_sc = forecast_sc.masked_fill(pad_mask.unsqueeze(1), 0.0)
            resonance_bias = (pos_score + forecast_sc).sum(dim=-1, keepdim=True)
        else:
            enriched = seq_expanded
            resonance_bias = 0.0

        pool_scores = self.pool_attn(enriched).squeeze(-1)
        pool_mask = pad_mask.unsqueeze(1).expand(-1, num_dst, -1)
        pool_scores = pool_scores.masked_fill(pool_mask, -1e9)
        pool_weights = torch.softmax(pool_scores, dim=-1).masked_fill(pool_mask, 0.0)

        src_pooled = torch.bmm(
            pool_weights.view(batch_size * num_dst, 1, -1),
            enriched.view(batch_size * num_dst, -1, self.hidden_size),
        ).view(batch_size, num_dst, self.hidden_size)

        match_product = src_pooled * dst_repr
        score_feat = torch.cat([src_pooled, dst_repr, match_product], dim=-1)
        output = self.score_proj(score_feat)
        output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
        output = self.output_layer(output.view(-1, output.shape[-1])).view(batch_size, num_dst, -1)
        output = output + resonance_bias
        return output

    def predict(
        self,
        src_neighb_seq,
        src_neighb_seq_len,
        src_neighb_interact_times,
        cur_pred_times,
        test_dst,
        dst_last_update_times,
        edge_ids=None,
        dst_neighb_seq=None,
        dst_neighb_interact_times=None,
    ):
        src_neighb_seq = src_neighb_seq.to(self.device) - self.dst_min_idx + 1
        test_dst = test_dst.to(self.device) - self.dst_min_idx + 1
        src_neighb_seq[src_neighb_seq < 0] = 0

        if dst_neighb_seq is not None:
            dst_neighb_seq = dst_neighb_seq.to(self.device) - self.dst_min_idx + 1
            dst_neighb_seq[dst_neighb_seq < 0] = 0
            dst_neighb_interact_times = dst_neighb_interact_times.to(self.device)

        edge_ids = edge_ids.to(self.device) if edge_ids is not None else None
        logits = self.forward(
            src_neighb_seq,
            src_neighb_seq_len.to(self.device),
            src_neighb_interact_times.to(self.device),
            cur_pred_times.to(self.device),
            test_dst=test_dst,
            dst_last_update_times=dst_last_update_times.to(self.device),
            edge_ids=edge_ids,
            dst_neighb_seq=dst_neighb_seq,
            dst_neighb_interact_times=dst_neighb_interact_times,
        )
        return logits[:, 0].sigmoid().flatten(), logits[:, 1:].sigmoid().flatten()

    def calculate_loss(
        self,
        src_neighb_seq,
        src_neighb_seq_len,
        src_neighb_interact_times,
        cur_pred_times,
        test_dst,
        dst_last_update_times,
        edge_ids=None,
        dst_neighb_seq=None,
        dst_neighb_interact_times=None,
    ):
        pos_prob, neg_prob = self.predict(
            src_neighb_seq,
            src_neighb_seq_len,
            src_neighb_interact_times,
            cur_pred_times,
            test_dst,
            dst_last_update_times,
            edge_ids=edge_ids,
            dst_neighb_seq=dst_neighb_seq,
            dst_neighb_interact_times=dst_neighb_interact_times,
        )
        batch_size = test_dst.shape[0]
        num_negs = test_dst.shape[1] - 1
        predicts = torch.cat([pos_prob, neg_prob], dim=0)
        labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size * num_negs)], dim=0).to(self.device)
        loss = self.loss_fct(predicts, labels)
        if hasattr(self, "aux_loss") and self.aux_loss != 0.0:
            loss = loss + self.stf_loss_weight * self.aux_loss
        return loss, predicts, labels


class _TCSDInteractionModel(_TCSDBase):
    def __init__(
        self,
        n_layers,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
        initializer_range,
        n_nodes,
        max_seq_length,
        device,
        loss_type,
        use_pos=True,
        input_cat_time_intervals=False,
        output_cat_time_intervals=False,
        output_cat_repeat_times=False,
        num_output_layer=1,
        emb_dropout_prob=0.1,
        skip_connection=False,
        latent_dim=32,
        kl_weight=0.05,
        num_latent_modes=4,
        use_structural_invariance=False,
        si_weight=0.1,
        edge_feat_dim=0,
    ):
        super().__init__(
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            initializer_range=initializer_range,
            n_nodes=n_nodes,
            max_seq_length=max_seq_length,
            device=device,
            loss_type=loss_type,
            use_pos=use_pos,
            input_cat_time_intervals=input_cat_time_intervals,
            output_cat_time_intervals=output_cat_time_intervals,
            output_cat_repeat_times=output_cat_repeat_times,
            num_output_layer=num_output_layer,
            emb_dropout_prob=emb_dropout_prob,
            skip_connection=skip_connection,
            latent_dim=latent_dim,
            kl_weight=kl_weight,
            num_latent_modes=num_latent_modes,
            use_structural_invariance=use_structural_invariance,
            si_weight=si_weight,
            edge_feat_dim=edge_feat_dim,
        )
        self.resonance_encoder = TemporalResonanceEncoder(hidden_size, num_features=4, dropout=hidden_dropout_prob)
        self.srv_forecaster = SRVForecaster(num_features=4, hidden_dim=32, dropout=hidden_dropout_prob)
        self.use_forecast = True
        self.use_resonance_bias = True

    def forward(
        self,
        src_neighb_seq,
        src_neighb_seq_len,
        neighbors_interact_times,
        cur_times,
        test_dst=None,
        dst_last_update_times=None,
        edge_ids=None,
        dst_neighb_seq=None,
        dst_neighb_interact_times=None,
    ):
        batch_size = src_neighb_seq.shape[0]
        num_dst = test_dst.shape[1]
        self.aux_loss = 0.0

        neighb_emb = self.node_embedding(src_neighb_seq)
        time_diffs = (cur_times.view(-1, 1) - neighbors_interact_times).to(self.device).float().clamp(min=0, max=1e10)
        time_feat = self.encode_time(time_diffs)
        pad_mask = src_neighb_seq == 0

        ranks = self.compute_temporal_ranks(time_diffs, pad_mask)
        rank_feat = self.rank_embedding(ranks.clamp(max=self.max_seq_length))

        if self.edge_feat_dim > 0 and hasattr(self, "edge_features_table") and self.edge_features_table is not None and edge_ids is not None:
            edge_feats = self.edge_features_table[edge_ids.long()]
            edge_feat_transformed = self.edge_feat_proj(edge_feats)
            input_emb = self.neighbor_time_proj(torch.cat([neighb_emb, time_feat, rank_feat, edge_feat_transformed], dim=-1))
        else:
            input_emb = self.neighbor_time_proj(torch.cat([neighb_emb, time_feat, rank_feat], dim=-1))

        if self.training and self.neighbor_drop_rate > 0:
            drop_mask = torch.rand(batch_size, src_neighb_seq.shape[1], device=self.device) < self.neighbor_drop_rate
            drop_mask = drop_mask & ~pad_mask
            all_dropped = (pad_mask | drop_mask).all(dim=1)
            if all_dropped.any():
                for idx in all_dropped.nonzero(as_tuple=True)[0]:
                    first_valid = (~pad_mask[idx]).nonzero(as_tuple=True)[0]
                    if len(first_valid) > 0:
                        drop_mask[idx, first_valid[0]] = False
            pad_mask = pad_mask | drop_mask

        input_emb = input_emb.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        seq_out = input_emb
        for seq_layer in self.sequence_layers:
            seq_out = seq_layer(seq_out, pad_mask)
            seq_out = torch.nan_to_num(seq_out, nan=0.0)
            seq_out = seq_out.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        active_float = (~pad_mask).unsqueeze(-1).float()
        src_ctx = (seq_out * active_float).sum(dim=1) / active_float.sum(dim=1).clamp(min=1.0)

        test_dst_emb = self.node_embedding(test_dst)
        dst_raw = test_dst_emb.view(batch_size, -1, self.hidden_size)
        dt_dst_raw = cur_times.view(-1, 1) - dst_last_update_times
        dt_dst_raw[dst_last_update_times < -1] = 0
        dt_dst_raw = dt_dst_raw.to(self.device).float().clamp(min=0, max=1e10)
        dst_time_enc = self.encode_time(dt_dst_raw)
        dst_time_ctx = self.dst_time_proj(dst_time_enc)

        dt_dst_neighb = None
        if dst_neighb_seq is not None:
            dst_neighb_emb_pool = self.node_embedding(dst_neighb_seq)
            dt_dst_neighb = cur_times.view(-1, 1, 1) - dst_neighb_interact_times
            dt_dst_neighb = dt_dst_neighb.to(self.device).float().clamp(min=0, max=1e10)
            dst_neighb_time_enc = self.encode_time(dt_dst_neighb)
            dst_neighb_pad = dst_neighb_seq == 0
            dst_pool = self.dst_pooler(src_ctx, dst_neighb_emb_pool, dst_neighb_time_enc, dst_neighb_pad)
            dst_repr = self.LayerNorm(dst_raw + dst_time_ctx + dst_pool)
        else:
            dst_repr = self.LayerNorm(dst_raw + dst_time_ctx)
        dst_repr = self.emb_dropout(dst_repr)

        seq_expanded = seq_out.unsqueeze(1).expand(-1, num_dst, -1, -1)
        if dst_neighb_seq is not None:
            src_norm = F.normalize(neighb_emb, dim=-1)
            dst_norm = F.normalize(dst_raw, dim=-1)
            w = torch.einsum("bsd,bnd->bns", src_norm, dst_norm).masked_fill(pad_mask.unsqueeze(1), 0.0)

            d = self._compute_bridge_score(
                src_neighb_seq,
                neighbors_interact_times.to(self.device),
                dst_neighb_seq,
                dst_neighb_interact_times,
                pad_mask,
            )

            src_ids = src_neighb_seq.unsqueeze(1)
            dst_ids = test_dst.unsqueeze(2)
            repeat_match = (src_ids == dst_ids) & (src_ids > 0)
            time_median = self._per_query_time_scale(time_diffs, pad_mask)
            time_weight = torch.exp(-time_diffs.clamp(min=0) / time_median).masked_fill(pad_mask, 0.0)
            r = repeat_match.float() * time_weight.unsqueeze(1)

            dst_neighb_emb = self.node_embedding(dst_neighb_seq)
            dst_nb_norm = F.normalize(dst_neighb_emb, dim=-1)
            cross_sim = torch.einsum("bsh,bdlh->bdsl", src_norm, dst_nb_norm)
            dst_pad = dst_neighb_seq == 0
            cross_sim = cross_sim.masked_fill(pad_mask.unsqueeze(1).unsqueeze(-1), -2.0)
            cross_sim = cross_sim.masked_fill(dst_pad.unsqueeze(2), -2.0)
            s, _ = cross_sim.max(dim=-1)
            s = s.clamp(min=0.0).masked_fill(pad_mask.unsqueeze(1), 0.0)

            srv = torch.stack([w, d, r, s], dim=-1)
            gate, pos_score = self.resonance_encoder(srv, pad_mask)
            forecast_sc = None
            if self.use_forecast:
                srv_pred, forecast_sc = self.srv_forecaster(srv, pad_mask)

            if self.use_forecast and self.training and srv.shape[2] > 1:
                pred_shift = srv_pred[:, :, :-1, :]
                target_shift = srv[:, :, 1:, :]
                valid_mask = ~pad_mask[:, 1:].unsqueeze(1).unsqueeze(-1)
                mse = ((pred_shift - target_shift) ** 2) * valid_mask.float()
                n_valid = valid_mask.float().sum().clamp(min=1.0)
                self.aux_loss = mse.sum() / (n_valid * srv.shape[1] * srv.shape[-1])

            enriched = seq_expanded * gate
            resonance_bias = 0.0
            if self.use_resonance_bias:
                pos_score = pos_score.masked_fill(pad_mask.unsqueeze(1), 0.0)
                resonance_bias = pos_score.sum(dim=-1, keepdim=True)
                if forecast_sc is not None:
                    forecast_sc = forecast_sc.masked_fill(pad_mask.unsqueeze(1), 0.0)
                    resonance_bias = resonance_bias + forecast_sc.sum(dim=-1, keepdim=True)
        else:
            enriched = seq_expanded
            resonance_bias = 0.0

        pool_scores = self.pool_attn(enriched).squeeze(-1)
        pool_mask = pad_mask.unsqueeze(1).expand(-1, num_dst, -1)
        pool_scores = pool_scores.masked_fill(pool_mask, -1e9)
        pool_weights = torch.softmax(pool_scores, dim=-1).masked_fill(pool_mask, 0.0)

        src_pooled = torch.bmm(
            pool_weights.view(batch_size * num_dst, 1, -1),
            enriched.view(batch_size * num_dst, -1, self.hidden_size),
        ).view(batch_size, num_dst, self.hidden_size)

        match_product = src_pooled * dst_repr
        score_feat = torch.cat([src_pooled, dst_repr, match_product], dim=-1)
        real_output = self.score_proj(score_feat)
        real_output = torch.nan_to_num(real_output, nan=0.0, posinf=1e6, neginf=-1e6)
        real_output = self.output_layer(real_output.view(-1, real_output.shape[-1])).view(batch_size, num_dst, -1)
        return real_output + resonance_bias

    def predict(
        self,
        src_neighb_seq,
        src_neighb_seq_len,
        src_neighb_interact_times,
        cur_pred_times,
        test_dst,
        dst_last_update_times,
        edge_ids=None,
        dst_neighb_seq=None,
        dst_neighb_interact_times=None,
    ):
        src_neighb_seq = src_neighb_seq.to(self.device) - self.dst_min_idx + 1
        test_dst = test_dst.to(self.device) - self.dst_min_idx + 1
        src_neighb_seq[src_neighb_seq < 0] = 0

        if dst_neighb_seq is not None:
            dst_neighb_seq = dst_neighb_seq.to(self.device) - self.dst_min_idx + 1
            dst_neighb_seq[dst_neighb_seq < 0] = 0
            dst_neighb_interact_times = dst_neighb_interact_times.to(self.device)

        edge_ids = edge_ids.to(self.device) if edge_ids is not None else None
        logits = self.forward(
            src_neighb_seq,
            src_neighb_seq_len.to(self.device),
            src_neighb_interact_times.to(self.device),
            cur_pred_times.to(self.device),
            test_dst=test_dst,
            dst_last_update_times=dst_last_update_times.to(self.device),
            edge_ids=edge_ids,
            dst_neighb_seq=dst_neighb_seq,
            dst_neighb_interact_times=dst_neighb_interact_times,
        )
        return logits[:, 0].sigmoid().flatten(), logits[:, 1:].sigmoid().flatten()

    def calculate_loss(
        self,
        src_neighb_seq,
        src_neighb_seq_len,
        src_neighb_interact_times,
        cur_pred_times,
        test_dst,
        dst_last_update_times,
        edge_ids=None,
        dst_neighb_seq=None,
        dst_neighb_interact_times=None,
    ):
        pos_prob, neg_prob = self.predict(
            src_neighb_seq,
            src_neighb_seq_len,
            src_neighb_interact_times,
            cur_pred_times,
            test_dst,
            dst_last_update_times,
            edge_ids=edge_ids,
            dst_neighb_seq=dst_neighb_seq,
            dst_neighb_interact_times=dst_neighb_interact_times,
        )
        batch_size = test_dst.shape[0]
        num_negs = test_dst.shape[1] - 1
        predicts = torch.cat([pos_prob, neg_prob], dim=0)
        labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size * num_negs)], dim=0).to(self.device)
        loss = self.loss_fct(predicts, labels)

        main_aux_loss = self.aux_loss if hasattr(self, "aux_loss") else 0.0
        if main_aux_loss != 0.0:
            loss = loss + self.stf_loss_weight * main_aux_loss

        return loss, predicts, labels


class TCSD(_TCSDInteractionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_forecast = False
        self.srv_forecaster = None
        self.use_resonance_bias = False




