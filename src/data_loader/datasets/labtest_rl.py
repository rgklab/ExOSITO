import os, sys, tarfile, collections, json, pickle
import torch
from os import listdir
import numpy as np
from torch.utils.data import Dataset
import pickle
import random
import time


class InvalidPatientException(Exception):
    pass


class PatientLabRLDataset(Dataset):
    def __init__(
        self,
        path,
        split="train",
        action_cost_coef=5e-5,
        action_cost=1,
        gamma=0.95,
        adj_reward=False,
    ):
        assert split in ["train", "val", "test"]
        self.path = os.path.join(path, f"rlexp_{split}.pkl")
        s = time.time()
        with open(self.path, "rb") as f:
            data_dict = pickle.load(f)
        e = time.time()
        print(f"load data took {e-s:.2f}")
        tmp_dict = dict()
        l = 0
        for k in data_dict:
            l = len(data_dict[k])
            break

        # for k in data_dict:
        #     tmp_dict[k] = data_dict[k][: int(l * 0.2)]
        # data_dict = tmp_dict
        print(data_dict.keys())
        for k in data_dict:
            print(l, len(data_dict[k]))
            break
        self.data = data_dict
        self.mortality = data_dict["mortality"]
        self.curr_states = data_dict["curr_states"]
        self.next_states = data_dict["next_states"]
        self.curr_history = data_dict["curr_history"]
        self.next_history = data_dict["next_history"]
        self.prob_gain = data_dict["prob_gain"]
        self.curr_prob = data_dict["curr_prob"]
        self.next_prob = data_dict["next_prob"]
        self.actions = data_dict["actions"]
        self.labels = data_dict["labels"]
        self.patient_inds = data_dict["sids"]
        # self.delay_update_state = data_dict["delay_update_state"]
        if "norm_feature" in data_dict:
            self.norm_feature = data_dict["norm_feature"]
        else:
            self.norm_feature = None
        self.split = split
        self.adj_reward = adj_reward

        self.gain_coef = 1
        self.action_cost_coef = action_cost_coef
        self.action_cost = action_cost
        self.gamma = gamma
        self.num_actions = 10

        print(type(self.actions))
        print(len(self.actions))
        print(len(self.curr_states))
        # print("===================check this dataset==================")

    def __len__(self):
        return len(self.curr_states)  # len(self.data["cur_states"])

    def __getitem__(self, idx):
        s, a, r, s_, gamma = self._get_exp(idx)
        # print(s.shape, a.shape, r.shape, gamma.shape, gamma)
        gamma = torch.FloatTensor(gamma)
        s = torch.FloatTensor(s)
        s_ = torch.FloatTensor(s_)
        a = torch.FloatTensor(a)
        r = torch.FloatTensor(r)
        return s, a, r, s_, gamma

    def _get_exp(self, idx):
        action = self.actions[idx]

        time_pass = False
        if action < 0:
            time_pass = True
        curr_state = self.curr_states[idx]
        next_state = self.next_states[idx]

        if self.norm_feature is not None:
            mean, std = self.norm_feature
            cur_state = (cur_state - mean) / std
            next_state = (next_state - mean) / std

        s = np.concatenate((curr_state, self.curr_history[idx]), axis=-1)
        s_ = np.concatenate((next_state, self.next_history[idx]), axis=-1)

        acts = [0] * (self.num_actions + 1)
        if action >= 0:
            acts[action] = 1
        else:
            acts[-1] = 1
        a = np.array(acts, dtype=np.int32)  # self.actions[idx]
        r = np.array(self._get_reward(idx), dtype=np.float)  # .reshape((1, 1))
        gamma = np.array(self._get_gamma(idx), dtype=np.float)  # .reshape((1, 1))
        # print(s.shape, a, r, s_.shape, gamma)
        return s, a, r, s_, gamma

    def _get_reward(self, idx):
        # If mortality is 0, then encourage to increase probability.
        # If mortality is 1, be negative.
        # info_gain = self.gain_coef * self.prob_gain[idx] * (-2 * self.labels[idx] + 1)
        action = self.actions[idx]
        time_pass = 0
        if action < 0:
            time_pass = 1
        info_gain = (
            self.gain_coef * self.prob_gain[idx] * (-2 * self.mortality[idx] + 1)
        )
        action_cost = self.action_cost * (1 - time_pass)  # changing action cost here
        r = info_gain - self.action_cost_coef * action_cost
        if time_pass and r == 0 and self.adj_reward:
            r += (2 * self.mortality[idx] - 1) * 0.1
        return r

    def _get_gamma(self, idx):
        action = self.actions[idx]
        time_pass = 0
        if action < 0:
            time_pass = 1
        return time_pass * self.gamma + (1 - time_pass)

    def get_label_weights(self):
        """
        sampler weights mortality
        """
        lb, cts = np.unique(self.mortality, return_counts=True)
        res = np.ones(len(self.mortality))
        self.mortality = np.array(self.mortality, dtype=np.int32)
        for l in lb:
            res[self.mortality == l] = res[self.mortality == l] / cts[int(l)]
        assert len(res) == len(self.mortality)
        return res

    def get_label_weights_l(self):
        """
        sampler weights labels
        """
        lb, cts = np.unique(self.labels, return_counts=True)
        res = np.ones(len(self.labels))
        self.labels = np.array(self.labels, dtype=np.int32)
        for l in lb:
            res[self.labels == l] = res[self.labels == l] / cts[int(l)]
        assert len(res) == len(self.labels)
        return res
