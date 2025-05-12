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
        from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction

        hfconfig = PatchTSMixerConfig(
            context_length=self.seq_len,
            prediction_length=self.pred_len,
            num_input_channels=configs.output_dim,
        )
        self.model = PatchTSMixerForPrediction(hfconfig)

        print("a")

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.model(x)
        x = x.prediction_outputs
        # print(type(x))
        # print(x.size())
        # x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x  # [Batch, Output length, Channel]


class PModel(nn.Module):
    """
    PatchTST from hugging face
    """

    def __init__(self, configs):
        super(PModel, self).__init__()
        self.seq_len = configs.x_len
        self.pred_len = configs.y_len
        from transformers import (
            PatchTSMixerConfig,
            PatchTSMixerForTimeSeriesClassification,
        )

        hfconfig = PatchTSMixerConfig(
            num_targets=configs.y_len,
            context_length=self.seq_len,
            prediction_length=self.pred_len,
            num_input_channels=configs.output_dim,
        )
        self.model = PatchTSMixerForTimeSeriesClassification(hfconfig)

        # print("a")

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.model(x)
        x = x.prediction_outputs
        x = torch.sigmoid(x)
        return x  # [Batch, Output length, Channel]


if __name__ == "__main__":
    print("patchtsmixer test")
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
    configs.x_len = 72
    configs.y_len = 10
    x = torch.zeros((100, 72, 71)).to(configs.device)
    model = PModel(configs).to(configs.device)
    y = model(x)
    print(y.size())
