import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch


class Model(nn.Module):
    """
    PatchTST from hugging face
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        from transformers import PatchTSTConfig, PatchTSTForPrediction

        hfconfig = PatchTSTConfig(
            context_length=self.seq_len,
            prediction_length=self.pred_len,
            num_input_channels=configs.output_dim,
        )
        self.model = PatchTSTForPrediction(hfconfig)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.model(x)
        x = x.prediction_outputs
        return x  # [Batch, Output length, Channel]


if __name__ == "__main__":
    print("patchtst test")
    import argparse

    parser = argparse.ArgumentParser(
        description="Lab test value Time Series Forecasting"
    )
    args = parser.parse_args()
    configs = args
    configs.seq_len = 48
    configs.hidden_dim = 128
    configs.layer_dim = 3
    configs.device = torch.device("cuda:1")
    configs.pred_len = 24
    configs.output_dim = 71
    x = torch.zeros((100, 48, 71)).to(configs.device)
    model = Model(configs).to(configs.device)
    y = model(x)
    print(y.size())
