import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    Just one Linear layer
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x  # [Batch, Output length, Channel]


class PModel(nn.Module):
    """
    Just one Linear layer
    """

    def __init__(self, configs):
        super(PModel, self).__init__()
        self.seq_len = configs.x_len
        self.pred_len = configs.y_len
        self.input_dim = configs.output_dim

        self.Linear = nn.Linear(self.seq_len * self.input_dim, self.pred_len)
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = x.view(x.size(0), self.seq_len * self.input_dim)
        x = self.Linear(x)
        x = torch.sigmoid(x)
        return x  # [Batch, Output length, Channel]
