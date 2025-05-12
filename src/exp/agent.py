import torch, os, sys, argparse
import torch.nn as nn
import numpy as np
from collections import deque
import random
from itertools import count
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
import random

from data_loader.datasets.labtest_rl import PatientLabRLDataset
from rl_exp.memory import Memory, TreeMemory
from rl_exp.agents import DQNAgent, CQLAgent, BCAgent, IQLAgent
from torch.utils.tensorboard import SummaryWriter


class OfflineRLTrainer:
    def __init__(self, config) -> None:
        print("initializing trainer")
        # device
        gpu_num = config.gpu_num
        print(f"gpu_num: {gpu_num}")
        self.device = torch.device(
            f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu"
        )
        # output path
        self.ds_id = config.dataset
        outdir = config.out_path
        if self.ds_id == "mimic3":
            print("use mimic3 outdir")
            time_stamp = outdir.split("_")[-1]
            outdir = "rl_exp3_" + time_stamp
        self.outdir = outdir


        self.n_action = 11
        self.hs_dim = 256
        self.init_eps = 0.2
        self.gamma = 0.95
        self.mem_size = config.mem_size
        self.bs = config.batch_size
        self.update_step = config.update_steps
        self.lr = config.lr
        self.epochs = config.epochs
        self.cost = config.action_cost
        self.cost_coef = config.action_cost_coef
        self.home_dir = config.home_dir
        self.model_str = config.model_str
        self.seed = config.seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # identifier
        ace = "{:.0e}".format(config.action_cost_coef)
        lrs = "{:.0e}".format(self.lr)
        s = "s" if config.sampler else "ns"
        other_params = f"{self.mem_size}_{self.update_step}_{s}_{self.seed}"
        unique_id = f"{self.bs}_{lrs}_{ace}_{self.cost}_{other_params}"
        outpath = os.path.join(
            self.home_dir, f"runs/labrls/{self.model_str}/{outdir}", unique_id
        )
        self.outpath = outpath

        print(outpath)

        # writer
        self.writer = SummaryWriter(outpath)
        # self.writer = None

        # dataset
        data_dir = "data/physionet/laborder"
        dataset = PatientLabRLDataset(
            data_dir,
            split="train",
            action_cost_coef=self.cost_coef,
            action_cost=self.cost,
            gamma=self.gamma,
        )
        val_dataset = PatientLabRLDataset(
            data_dir,
            split="val",
            action_cost_coef=self.cost_coef,
            action_cost=self.cost,
            gamma=self.gamma,
        )

        # print(dataset[14])

        # sampler
        wts = dataset.get_label_weights()
        # wts = dataset.get_label_weights_l()
        # print(wts.shape)
        if config.sampler:
            sampler = WeightedRandomSampler(
                weights=list(wts), num_samples=len(dataset), replacement=True
            )
        else:
            sampler = None
        shuffle = True
        if sampler:
            shuffle = False

        # loader
        self.train_loader = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=self.bs,
            num_workers=8,
            shuffle=shuffle,
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            sampler=None,
            batch_size=self.bs,
            num_workers=8,
            shuffle=False,
        )


    def train(self):
        """
        train with various agent
        """
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)

        train_loader, val_loader = self.train_loader, self.val_loader
        memory_replay = TreeMemory(self.mem_size)
        writer = self.writer
        device = self.device

        if self.model_str == "ddqn":
            agent = DQNAgent(
                self.hs_dim + self.n_action - 1,
                self.n_action,
                self.lr,
                writer,
                tmax=len(train_loader),
                device=device,
                update_step=self.update_step,
            )
        elif self.model_str == "bc":
            agent = BCAgent(
                self.hs_dim + self.n_action - 1,
                self.n_action,
                self.lr,
                writer,
                tmax=len(train_loader),
                device=device,
                update_step=self.update_step,
            )
        elif self.model_str == "cql":
            agent = CQLAgent(
                self.hs_dim + self.n_action - 1,
                self.n_action,
                self.lr,
                writer,
                tmax=len(train_loader),
                device=device,
                update_step=self.update_step,
            )
        elif self.model_str == "iql":
            agent = IQLAgent(
                self.hs_dim + self.n_action - 1,
                self.n_action,
                self.lr,
                writer,
                tmax=len(train_loader),
                device=device,
                update_step=self.update_step,
            )
        else:
            raise NotImplementedError

        # model
        max_epochs = self.epochs
        best_val_loss = np.inf
        for epoch in range(max_epochs):
            episode_reward = 0
            tbar = tqdm(train_loader)
            # train loop
            for batch_idx, exp in enumerate(tbar):
                s, a, r, s_, gamma = exp
                # add into memory
                bs = int(s.size()[0])
                for idx in range(bs):
                    one_exp = (s[idx], a[idx], r[idx], s_[idx], gamma[idx])
                    episode_reward += r[idx]
                    memory_replay.add(one_exp)
                # counter += bs
                if memory_replay.size() < memory_replay.memory_size:
                    continue

                midx, batch, mweights = memory_replay.sample(bs, False)
                # learn_steps += 1
                # learning with agent
                loss, updated_memory = agent.learn(batch, memory_replay, midx, mweights)
                memory_replay = updated_memory

            # val loop
            print(f"Validation epoch {epoch}")
            with torch.no_grad():
                agent.network.eval()
                val_loss = 0.0
                ct = 0
                vbar = tqdm(val_loader)
                for idx, exp in enumerate(vbar):
                    loss = agent.val_eval(exp)
                    val_loss += loss.item()
                    ct += 1
                val_loss /= ct
            agent.network.train()

            self.writer.add_scalar("loss/val_loss_per_ep", val_loss, global_step=epoch)
            if val_loss < best_val_loss:
                print(
                    f"Update weights, from {round(best_val_loss,3)} to {round(val_loss,3)}"
                )
                best_val_loss = val_loss
                self.writer.add_scalar(
                    "loss/best_val_loss_by_ep", best_val_loss, global_step=epoch
                )
                torch.save(
                    agent.network.state_dict(),
                    os.path.join(self.outpath, f"{self.model_str}-policy.pt"),
                )
            if epoch % 1 == 0:
                torch.save(
                    agent.network.state_dict(),
                    os.path.join(self.outpath, f"last-{self.model_str}-policy.pt"),
                )
                print(
                    "Ep {}\tMoving average score: {:.2f}\t".format(
                        epoch, episode_reward.item()
                    )
                )
            agent.scheduler.step()
            self.writer.add_scalar(
                "misc/cosine_lr_decay",
                agent.scheduler.get_last_lr()[0],
                global_step=agent.learn_steps,
            )
            writer.add_scalar("episode reward", episode_reward, global_step=epoch)

        return
