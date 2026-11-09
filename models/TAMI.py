"""
TAMI baseline: a historical decoder with a memory of the interactions of each pair, on top of a dynamic graph backbone.
Z. Yu et al., TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction, NeurIPS 2025.
"""
import copy

import torch
import torch.nn as nn


class HistEmbAggregatorWeightedSum(nn.Module):
    """
    Update historical edge embeddings by exponentially smoothing the current
    pair representation with the stored edge memory.
    """

    def __init__(self, gamma: float = 0.9):
        super().__init__()
        self.gamma = gamma
        self.aggregator_name = "weighted_sum"

    def forward(self, current_emb: torch.Tensor, hist_emb: torch.Tensor) -> torch.Tensor:
        return self.gamma * current_emb + (1.0 - self.gamma) * hist_emb


class TRCMemory(nn.Module):
    """
    Temporal Relation Context memory keyed by (src_id, dst_id).
    """

    def __init__(self, dim: int, device: str):
        super().__init__()
        self.device = device
        self.PAD_ZERO = torch.zeros(dim, device=device, requires_grad=False)
        self.most_recent_hist_emb = {}

    def update_memories(self, keys, hist_emb: torch.Tensor) -> None:
        for idx, key in enumerate(keys):
            self.most_recent_hist_emb[key] = hist_emb[idx]

    def get_memories(self, keys):
        return [
            self.PAD_ZERO if key not in self.most_recent_hist_emb else self.most_recent_hist_emb[key].to(self.device)
            for key in keys
        ]

    def reset_memory(self) -> None:
        self.most_recent_hist_emb.clear()

    def backup_memory_bank(self):
        return {"most_recent_emb": copy.deepcopy(self.most_recent_hist_emb)}

    def load_memory_bank(self, bank) -> None:
        self.reset_memory()
        self.most_recent_hist_emb = bank["most_recent_emb"]

    def reload_memory_bank(self, bank) -> None:
        self.load_memory_bank(bank)


class HistoricalDecoder(nn.Module):
    """
    TAMI historical decoder that augments pair scoring with edge-level temporal
    relation memory.
    """

    def __init__(
        self,
        input_dim1: int,
        input_dim2: int,
        hidden_dim: int,
        output_dim: int,
        device: str = "cpu",
        gamma: float = 0.9,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim * 2, output_dim)
        self.act = nn.ReLU()
        self.device = device
        self.aggregate_hist_emb = HistEmbAggregatorWeightedSum(gamma=gamma)
        self.hist_emb_aggregator = self.aggregate_hist_emb.aggregator_name
        self.historical_interaction_memory = TRCMemory(dim=input_dim1, device=device)

    def forward(
        self,
        src_ids,
        dst_ids,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
        update_memories: bool = False,
    ) -> torch.Tensor:
        x = torch.cat([src_emb, dst_emb], dim=1)
        h_current = self.act(self.fc1(x))

        keys = list(zip(src_ids.tolist() if hasattr(src_ids, "tolist") else src_ids,
                        dst_ids.tolist() if hasattr(dst_ids, "tolist") else dst_ids))
        hist_emb_most_recent_one = self.historical_interaction_memory.get_memories(keys)
        h_most_recent_emb = torch.stack(hist_emb_most_recent_one, dim=0).to(self.device)

        h_historical = self.aggregate_hist_emb(current_emb=h_current, hist_emb=h_most_recent_emb)
        if self.hist_emb_aggregator != "weighted_sum":
            raise ValueError("Invalid TRC aggregator")
        updated_weighted_hist_emb = h_historical
        h_historical = h_most_recent_emb

        h = self.fc2(torch.cat((h_current, h_historical), dim=1))

        if update_memories:
            self.historical_interaction_memory.update_memories(keys, updated_weighted_hist_emb.detach().cpu())
        return h


def score_pairs(model, model_name: str, src_node_ids, dst_node_ids, src_node_embeddings, dst_node_embeddings, update_memories: bool = False):
    """
    score node pairs with the link predictor model[1]; the TAMI decoder also needs the node ids to read and write its memory of each pair
    :param update_memories: bool, whether TAMI stores these pairs in its memory, done for the positive pairs
    :return: Tensor, shape (*, 1), logits
    """
    if model_name == 'TAMI':
        return model[1](src_ids=src_node_ids, dst_ids=dst_node_ids, src_emb=src_node_embeddings, dst_emb=dst_node_embeddings, update_memories=update_memories)
    return model[1](input_1=src_node_embeddings, input_2=dst_node_embeddings)
