import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random
from models.ddqn import DDQNetwork, Value


def iqlval_loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class DQNAgent:
    def __init__(
        self, state_dim, action_dim, lr, writer, tmax=-1, device="cpu", update_step=1000
    ):
        self.device = device
        self.writer = writer
        self.n_action = action_dim
        self.update_step = update_step
        self.target_net = DDQNetwork(state_dim=state_dim, action_dim=action_dim).to(
            self.device
        )
        self.network = DDQNetwork(state_dim=state_dim, action_dim=action_dim).to(
            self.device
        )

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=tmax, eta_min=0, last_epoch=-1
        )
        self.tau = 1e-3
        self.gamma = 0.95
        self.learn_steps = 0

    def learn(self, batch, memory_replay, midx, mweights):
        if self.learn_steps % self.update_step == 0:
            self.target_net.load_state_dict(self.network.state_dict())
        device = self.device
        s, a, r, s_, gamma = zip(*batch)
        s = torch.stack(s, dim=0).to(device)
        a = torch.stack(a, dim=0).to(device)
        r = torch.stack(r, dim=0).to(device)
        s_ = torch.stack(s_, dim=0).to(device)
        gamma = torch.stack(gamma, dim=0).to(device)
        if r.dim() == 1:
            r = r.unsqueeze(1)
        if gamma.dim() == 1:
            gamma = gamma.unsqueeze(1)
        # print(s.size(), r.size())
        self.learn_steps += 1
        with torch.no_grad():
            onlineQ_next = self.network(s_)
            online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
            # online_max_action = torch.argmax(adj_q, dim=1, keepdim=True)
            targetQ_next = self.target_net(s_)
            adj_q = self._get_adj_target_q_val(s, targetQ_next)
            y = r + gamma * adj_q.gather(1, online_max_action.long())
            # print(adj_q.gather(1, online_max_action.long()).size())
        # print(adj_q.size(), online_max_action.size(), r.size(), gamma.size(), y.size())
        pred = self.network(s)
        real_a = torch.argmax(a, dim=1, keepdim=True).long()
        qval = pred.gather(1, real_a.long())

        # print(y.size(), qval.size())
        if len(mweights) > 0:
            nrloss = F.mse_loss(qval, y, reduction="none")
            memory_replay.batch_update(
                midx, nrloss.cpu().detach().numpy()
            )  # update tree idx
            loss = (torch.Tensor(mweights).to(device) * nrloss).mean()
        else:
            loss = F.mse_loss(qval, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.writer.add_scalar(
            "loss/train_loss", loss.item(), global_step=self.learn_steps
        )

        return loss, memory_replay

    def val_eval(self, exp):
        device = self.device
        s, a, r, s_, gamma = exp
        s = s.to(device)
        a = a.to(device)
        r = r.to(device)
        s_ = s_.to(device)
        gamma = gamma.to(device)
        if r.dim() == 1:
            r = r.unsqueeze(1)
        if gamma.dim() == 1:
            gamma = gamma.unsqueeze(1)

        onlineQ_next = self.network(s_)
        targetQ_next = self.target_net(s_)
        online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
        adj_q = self._get_adj_target_q_val(s, targetQ_next)
        # online_max_action = torch.argmax(adj_q, dim=1, keepdim=True)
        # print(adj_q.size())
        y = r + gamma * adj_q.gather(1, online_max_action.long())

        pred = self.network(s)
        real_a = torch.argmax(a, dim=1, keepdim=True).long()
        qval = pred.gather(1, real_a.long())
        loss = F.mse_loss(qval, y)
        return loss

    def _get_adj_target_q_val(self, s, q_next):
        # get history action
        pad = torch.zeros((s.size()[0]), 1, dtype=s.dtype, device=self.device)
        raw_history_action = s[:, -(self.n_action - 1) :]
        his_act = torch.concat([raw_history_action, pad], axis=1)
        # print(his_act.size())
        min_q = torch.min(q_next, axis=-1, keepdim=True)[0]
        adj_q = his_act * (min_q - 1) + (1 - his_act) * q_next
        return adj_q


class CQLAgent:
    def __init__(
        self, state_dim, action_dim, lr, writer, tmax=-1, device="cpu", update_step=1000
    ):
        self.device = device
        self.writer = writer
        self.n_action = action_dim
        self.update_step = update_step
        self.target_net = DDQNetwork(state_dim=state_dim, action_dim=action_dim).to(
            self.device
        )
        self.network = DDQNetwork(state_dim=state_dim, action_dim=action_dim).to(
            self.device
        )

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=tmax, eta_min=0, last_epoch=-1
        )
        self.tau = 1e-3
        self.gamma = 0.95
        self.learn_steps = 0

    def get_action(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.network.eval()
            with torch.no_grad():
                action_values = self.network(state)
            self.network.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
        else:
            action = random.choices(np.arange(self.action_size), k=1)
        return action

    def learn(self, batch, memory_replay, midx, mweights):
        if self.learn_steps % self.update_step == 0:
            self.soft_update(self.network, self.target_net)

        device = self.device
        s, a, r, s_, gamma = zip(*batch)
        s = torch.stack(s, dim=0).to(device)
        a = torch.stack(a, dim=0).to(device)
        r = torch.stack(r, dim=0).to(device)
        s_ = torch.stack(s_, dim=0).to(device)
        gamma = torch.stack(gamma, dim=0).to(device)

        self.learn_steps += 1
        with torch.no_grad():
            onlineQ_next = self.network(s_)
            online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
            # online_max_action = torch.argmax(adj_q, dim=1, keepdim=True)
            targetQ_next = self.target_net(s_)
            adj_q = self._get_adj_target_q_val(s, targetQ_next)
            y = r + gamma * adj_q.gather(1, online_max_action.long())

        pred = self.network(s)
        real_a = torch.argmax(a, dim=1, keepdim=True).long()
        qval = pred.gather(1, real_a.long())

        cql1_loss = self.cql_loss(pred, real_a.long())
        if len(mweights) > 0:
            nrloss = F.mse_loss(qval, y, reduction="none")
            memory_replay.batch_update(
                midx, nrloss.cpu().detach().numpy()
            )  # update tree idx
            loss = cql1_loss + 0.5 * (torch.Tensor(mweights).to(device) * nrloss).mean()
        else:
            loss = cql1_loss + 0.5 * F.mse_loss(qval, y)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        self.writer.add_scalar(
            "loss/train_loss", loss.item(), global_step=self.learn_steps
        )
        return loss, memory_replay

    def val_eval(self, exp):
        device = self.device
        s, a, r, s_, gamma = exp
        s = s.to(device)
        a = a.to(device)
        r = r.to(device)
        s_ = s_.to(device)
        gamma = gamma.to(device)

        onlineQ_next = self.network(s_)
        targetQ_next = self.target_net(s_)
        online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
        adj_q = self._get_adj_target_q_val(s, targetQ_next)
        # online_max_action = torch.argmax(adj_q, dim=1, keepdim=True)
        # print(adj_q.size())
        y = r + gamma * adj_q.gather(1, online_max_action.long())

        pred = self.network(s)
        real_a = torch.argmax(a, dim=1, keepdim=True).long()
        qval = pred.gather(1, real_a.long())
        cql1_loss = self.cql_loss(pred, real_a.long())
        loss = cql1_loss + 0.5 * F.mse_loss(qval, y)
        return loss

    def cql_loss(self, q_values, current_action):
        """Computes the CQL loss for a batch of Q-values and actions."""
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, current_action)

        return (logsumexp - q_a).mean()

    def _get_adj_target_q_val(self, s, q_next):
        # get history action
        pad = torch.zeros((s.size()[0]), 1, dtype=s.dtype, device=self.device)
        raw_history_action = s[:, -(self.n_action - 1) :]
        his_act = torch.concat([raw_history_action, pad], axis=1)
        # print(his_act.size())
        min_q = torch.min(q_next, axis=-1, keepdim=True)[0]
        adj_q = his_act * (min_q - 1) + (1 - his_act) * q_next
        return adj_q

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )


class IQLAgent:
    def __init__(
        self, state_dim, action_dim, lr, writer, tmax=-1, device="cpu", update_step=1000
    ):
        self.device = device
        self.writer = writer
        self.n_action = action_dim
        self.update_step = update_step
        self.target_net = DDQNetwork(state_dim=state_dim, action_dim=action_dim).to(
            self.device
        )
        self.network = DDQNetwork(state_dim=state_dim, action_dim=action_dim).to(
            self.device
        )

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=tmax, eta_min=0, last_epoch=-1
        )
        self.tau = 1e-3
        self.gamma = 0.95
        self.learn_steps = 0

        self.expectile = torch.FloatTensor([0.8]).to(device)
        self.value_net = Value(state_dim).to(self.device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

    def learn(self, batch, memory_replay, midx, mweights):
        if self.learn_steps % self.update_step == 0:
            self.soft_update(self.network, self.target_net)

        if self.learn_steps % 10 == 0:
            self.target_net.load_state_dict(self.network.state_dict())

        device = self.device
        s, a, r, s_, gamma = zip(*batch)
        s = torch.stack(s, dim=0).to(device)
        a = torch.stack(a, dim=0).to(device)
        r = torch.stack(r, dim=0).to(device)
        s_ = torch.stack(s_, dim=0).to(device)
        gamma = torch.stack(gamma, dim=0).to(device)

        self.learn_steps += 1

        self.value_optimizer.zero_grad()
        value_loss = self.calc_value_loss(s, a)
        value_loss.backward()
        self.value_optimizer.step()

        with torch.no_grad():
            onlineQ_next = self.network(s_)
            online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
            # online_max_action = torch.argmax(adj_q, dim=1, keepdim=True)
            targetQ_next = self.value_net(s_)
            adj_q = self._get_adj_target_q_val(s, targetQ_next)
            y = r + gamma * adj_q.gather(1, online_max_action.long())

        pred = self.network(s)
        real_a = torch.argmax(a, dim=1, keepdim=True).long()
        qval = pred.gather(1, real_a.long())
        if len(mweights) > 0:
            nrloss = F.mse_loss(qval, y, reduction="none")
            memory_replay.batch_update(
                midx, nrloss.cpu().detach().numpy()
            )  # update tree idx
            loss = (torch.Tensor(mweights).to(device) * nrloss).mean()
        else:
            loss = F.mse_loss(qval, y)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        self.writer.add_scalar(
            "loss/train_loss", loss.item(), global_step=self.learn_steps
        )
        return loss, memory_replay

    def val_eval(self, exp):
        device = self.device
        s, a, r, s_, gamma = exp
        s = s.to(device)
        a = a.to(device)
        r = r.to(device)
        s_ = s_.to(device)
        gamma = gamma.to(device)

        onlineQ_next = self.network(s_)
        targetQ_next = self.value_net(s_)
        online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
        adj_q = self._get_adj_target_q_val(s, targetQ_next)
        # online_max_action = torch.argmax(adj_q, dim=1, keepdim=True)
        # print(adj_q.size())
        y = r + gamma * adj_q.gather(1, online_max_action.long())

        pred = self.network(s)
        real_a = torch.argmax(a, dim=1, keepdim=True).long()
        qval = pred.gather(1, real_a.long())
        loss = F.mse_loss(qval, y)
        return loss

    def calc_value_loss(self, states, actions):
        with torch.no_grad():
            real_a = torch.argmax(actions, dim=1, keepdim=True).long()
            q = self.target_net(states).gather(1, real_a.long())

        value = self.value_net(states)
        value_loss = iqlval_loss(q - value, self.expectile).mean()
        return value_loss

    def _get_adj_target_q_val(self, s, q_next):
        # get history action
        pad = torch.zeros((s.size()[0]), 1, dtype=s.dtype, device=self.device)
        raw_history_action = s[:, -(self.n_action - 1) :]
        his_act = torch.concat([raw_history_action, pad], axis=1)
        # print(his_act.size())
        min_q = torch.min(q_next, axis=-1, keepdim=True)[0]
        adj_q = his_act * (min_q - 1) + (1 - his_act) * q_next
        return adj_q

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )


class BCAgent:
    def __init__(
        self, state_dim, action_dim, lr, writer, tmax=-1, device="cpu", update_step=1000
    ):
        self.device = device
        self.writer = writer
        self.n_action = action_dim
        self.update_step = update_step
        # self.target_net = DDQNetwork(state_dim=state_dim, action_dim=action_dim).to(
        #     self.device
        # )
        self.network = DDQNetwork(state_dim=state_dim, action_dim=action_dim).to(
            self.device
        )

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=tmax, eta_min=0, last_epoch=-1
        )
        self.tau = 1e-3
        self.gamma = 0.95
        self.learn_steps = 0

    def learn(self, batch, memory_replay, midx, mweights):
        # if self.learn_steps % self.update_step == 0:
        #     self.target_net.load_state_dict(self.network.state_dict())
        device = self.device
        s, a, r, s_, gamma = zip(*batch)
        s = torch.stack(s, dim=0).to(device)
        a = torch.stack(a, dim=0).to(device)
        r = torch.stack(r, dim=0).to(device)
        s_ = torch.stack(s_, dim=0).to(device)
        gamma = torch.stack(gamma, dim=0).to(device)

        self.learn_steps += 1

        pred = self.network(s)
        loss = F.cross_entropy(pred, a)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.writer.add_scalar(
            "loss/train_loss", loss.item(), global_step=self.learn_steps
        )

        return loss, memory_replay

    def val_eval(self, exp):
        device = self.device
        s, a, r, s_, gamma = exp
        s = s.to(device)
        a = a.to(device)
        r = r.to(device)
        s_ = s_.to(device)
        gamma = gamma.to(device)


        pred = self.network(s)
        loss = F.cross_entropy(pred, a)
        return loss
