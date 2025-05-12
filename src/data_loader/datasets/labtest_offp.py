import os, sys, tarfile, collections, json, pickle, time, math
import torch
from os import listdir
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import random

from itertools import product


def generate_configurations(x, y, n):
    # Ensure y is valid with respect to x
    y = [max(x[i], y[i]) for i in range(len(x))]

    # Find indices where y is 1 and x is 0
    variable_indices = [i for i in range(len(x)) if x[i] == 0 and y[i] == 1]

    # Generate all combinations for these indices
    all_combinations = []
    for i in range(
        1, 2 ** len(variable_indices) - 1
    ):  # Exclude 0 and max to avoid x and y
        new_combo = x.copy()

        # Set variable indices based on the current combination number
        for j, index in enumerate(variable_indices):
            if i & (1 << j):
                new_combo[index] = 1

        all_combinations.append(new_combo)

    # Randomize the order of combinations
    random.shuffle(all_combinations)

    # Limit to n combinations if there are more than n
    return all_combinations[:n]


class LabPiDatasetMIMIC(Dataset):
    """Dataset for lab test order (with bound) for MIMIC-IV"""

    def __init__(self, path=None, split="train", data_dict=None):
        print("load cached dataset")
        self.path = path
        self.split = split
        assert self.split in ["train", "val", "test"]
        assert self.split in path
        s = time.time()
        if data_dict is None:
            with open(self.path, "rb") as f:
                data_dict = pickle.load(f)
        else:
            data_dict = data_dict
        e = time.time()
        print(f"Load data took: {e-s:.3f} s")

        self.num_tsamples = 8

        # for k in data_dict:
        #     print(k, len(data_dict[k]))
        self.sids = data_dict["sids"]

        self.xs = data_dict["xs"]
        self.pred_ys = data_dict["pred_ys"]
        self.true_ys = data_dict["true_ys"]
        self.rule_orders = data_dict["rule_orders"]
        self.up_orders = data_dict["up_orders"]
        self.real_orders = data_dict["real_orders"]
        self.rulexy_orders = data_dict["rulexy_orders"]

        self.labidx_map = data_dict["labidx_map"]
        self.t2f_map = data_dict["t2f_map"]
        self.f2t_map = data_dict["f2t_map"]
        self.t2i_map = data_dict["t2i_map"]
        self.i2t_map = data_dict["i2t_map"]
        self.tests = data_dict["tests"]

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.xs[idx]).float()
        true_y = torch.from_numpy(self.true_ys[idx]).float()
        pred_y = torch.from_numpy(self.pred_ys[idx]).float()
        lower = torch.from_numpy(self.rule_orders[idx]).float()
        lowerxy = torch.from_numpy(self.rulexy_orders[idx]).float()
        t = torch.from_numpy(self.real_orders[idx]).float()
        upper = torch.from_numpy(self.up_orders[idx]).float()

        return x, true_y, pred_y, lower, lowerxy, t, upper


class LabGPSDatasetMIMIC(Dataset):
    """Dataset for lab test order (with bound) for MIMIC-IV"""

    def __init__(self, path=None, split="train", data_dict=None):
        print("load cached dataset")

        self.path = path
        self.split = split

        assert self.split in ["train", "val", "test", "all"]
        assert self.split in path
        s = time.time()
        if data_dict is None:
            with open(self.path, "rb") as f:
                data_dict = pickle.load(f)
        else:
            data_dict = data_dict
        e = time.time()
        print(f"Load data took: {e-s:.3f} s")

        self.num_tsamples = 8

        # for k in data_dict:
        #     print(k, len(data_dict[k]))
        self.sids = data_dict["sids"]
        self.xs = data_dict["xs"]
        self.pred_ys = data_dict["pred_ys"]
        self.true_ys = data_dict["true_ys"]
        self.rule_orders = data_dict["rule_orders"]
        self.up_orders = data_dict["up_orders"]
        self.real_orders = data_dict["real_orders"]
        self.rulexy_orders = data_dict["rulexy_orders"]

        self._proc_dataset()

    def _proc_dataset(self):
        self.total_idx = []
        lows = self.rule_orders
        ups = self.up_orders
        self.all_ts = []
        for i, (low, up) in enumerate(zip(lows, ups)):
            if list(low) != list(up):
                # print(low, up)
                ts = generate_configurations(low, up, self.num_tsamples)
                # print(len(ts))
                # print(ts)
                for t in ts:
                    self.all_ts.append(t)
                    self.total_idx.append(i)
                self.all_ts.append(low)
                self.total_idx.append(i)
                self.all_ts.append(up)
                self.total_idx.append(i)
                # break
            else:
                self.all_ts.append(up)
                self.total_idx.append(i)
        assert len(self.total_idx) == len(self.all_ts)
        return

    def __len__(self):
        return len(self.total_idx)

    def __getitem__(self, idx):
        oriidx = self.total_idx[idx]
        t = self.all_ts[idx]
        x = torch.from_numpy(self.xs[oriidx]).float()
        pred_y = torch.from_numpy(self.pred_ys[oriidx]).float()
        true_y = torch.from_numpy(self.true_ys[oriidx]).float()
        t = torch.from_numpy(t).float()

        return x, true_y, pred_y, t
