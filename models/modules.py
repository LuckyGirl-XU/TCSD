import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import copy


import math


import torch.nn.functional as fn


class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score, mask=None):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score))
        if mask is not None:
            loss = loss[mask]
        return loss.mean()


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, pos_score, neg_scores_2d):
        pos = pos_score.unsqueeze(1) / self.temperature
        neg = neg_scores_2d / self.temperature
        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


class MergeLayer(nn.Module):

    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int):
        """
        Merge Layer to merge two inputs via: input_dim1 + input_dim2 -> hidden_dim -> output_dim.
        :param input_dim1: int, dimension of first input
        :param input_dim2: int, dimension of the second input
        :param hidden_dim: int, hidden dimension
        :param output_dim: int, dimension of the output
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        """
        merge and project the inputs
        :param input_1: Tensor, shape (*, input_dim1)
        :param input_2: Tensor, shape (*, input_dim2)
        :return:
        """
        # Tensor, shape (*, input_dim1 + input_dim2)
        x = torch.cat([input_1, input_2], dim=-1)
        # Tensor, shape (*, output_dim)
        h = self.fc2(self.act(self.fc1(x)))
        return h


class MulMergeLayer(nn.Module):

    def __init__(self,scale: int =12):
        super().__init__()
        self.scale=scale

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        scores=torch.mul(input_1,input_2).sum(1)*self.scale
        return scores


