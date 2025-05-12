import torch, os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import WeightedRandomSampler


class MortalityLoader(DataLoader):
    """
    Data loader for time series for mortality prediction (binary classification)
    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        sampler=None,
        num_workers=8,
        collate_fn=default_collate,
    ):
        self.shuffle = shuffle
        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.sampler = sampler

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            # "collate_fn": collate_fn,
            "num_workers": num_workers,
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)


class PredictionLoader(DataLoader):
    """
    Data loader for time series for value prediction
    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        sampler=None,
        num_workers=8,
        collate_fn=default_collate,
    ):
        self.shuffle = shuffle
        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.sampler = sampler

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            # "collate_fn": collate_fn,
            "num_workers": num_workers,
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)
