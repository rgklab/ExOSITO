import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
import numpy as np


class LSTM_Model(nn.Module):
    def __init__(self, device, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM_Model, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        out = torch.sigmoid(out)
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.to(self.device) for t in (h0, c0)]

    def get_hidden(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))

        return hn, cn


class LSTM_CLS(nn.Module):
    def __init__(self, args):
        super(LSTM_CLS, self).__init__()
        # device, input_dim, hidden_dim, layer_dim, output_dim
        self.device = args.device
        self.hidden_dim = args.hidden_dim
        self.layer_dim = args.layer_dim
        self.input_dim = args.output_dim
        self.hidden_dim = args.hidden_dim
        self.rnn = nn.LSTM(
            self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True
        )
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        out = torch.sigmoid(out)
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.to(self.device) for t in (h0, c0)]

    def get_hidden(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        return hn, cn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.input_dim = configs.output_dim
        self.hidden_dim = configs.hidden_dim
        self.layer_dim = configs.layer_dim
        self.device = configs.device
        self.pred_len = configs.pred_len
        self.output_dim = configs.output_dim
        self.rnn = nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            self.layer_dim,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)



    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (h, c) = self.rnn(x, (h0, c0))
        pred = out[:, : self.pred_len, :]
        pred = self.fc(pred)

        return pred

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.to(self.device) for t in (h0, c0)]

    def get_hidden(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))

        return hn, cn


if __name__ == "__main__":
    print("lstm test")
    # import argparse

    # parser = argparse.ArgumentParser(
    #     description="Lab test value Time Series Forecasting"
    # )
    # args = parser.parse_args()
    # configs = args
    # configs.seq_len = 48
    # configs.hidden_dim = 128
    # configs.layer_dim = 3
    # configs.device = torch.device("cuda:2")
    # configs.pred_len = 24
    # configs.output_dim = 71
    # x = torch.zeros((100, 48, 71)).to(configs.device)
    # model = Model(configs).to(configs.device)
    # y = model(x)
    # print(y.size())
