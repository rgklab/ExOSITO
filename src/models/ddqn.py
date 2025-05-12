import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
import numpy as np


class DDQNetwork(nn.Module):
    def __init__(self, device=None, state_dim=294, action_dim=39):
        super(DDQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, 64)
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(64, 256)
        self.fc_adv = nn.Linear(64, 256)

        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, action_dim)
        self.action_dim = action_dim

    def forward(self, state):
        y = self.relu(self.fc1(state))
        value = self.relu(self.fc_value(y))
        adv = self.relu(self.fc_adv(y))

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage

        return Q  # bs x act_dim

    def select_action(self, state, old_acts):
        with torch.no_grad():
            Q = self.forward(state)
            Q[torch.from_numpy(old_acts)] = -999.0

            # print(Q)
            action_index = torch.argmax(Q, dim=1)

        return action_index

    def get_best_next_actions(self, state):
        cur_state = state.clone().detach().cpu().numpy()
        bs = state.size()[0]
        # cur_history = np.zeros((bs, self.action_dim - 1))
        agent_a = np.zeros((bs, self.action_dim))
        agent_a[:, :-1] = cur_state[:, 256:]
        # cur_state[:, 256:] = cur_history
        s = torch.from_numpy(cur_state).to(state.device, state.dtype)
        old_acts = agent_a != 0  # np.zeros((bs, self.action_dim), dtype=bool)
        acts = self.select_action(s, old_acts)
        acts = acts.cpu().numpy()
        agent_a[:, acts] = 1
        return agent_a

    def get_best_seq_actions(self, state):
        """
        get the sequential actions for the labtests
        given the patient, generate history along the way
        until hit the final action
        """
        # turn this tensor into something else
        cur_state = state.clone().detach().cpu().numpy()
        bs = state.size()[0]
        cur_history = np.zeros((bs, self.action_dim - 1))
        agent_a = np.zeros((bs, self.action_dim))
        # Start with empty history

        kept = np.ones((bs,), dtype=bool)
        old_acts = np.zeros((bs, self.action_dim), dtype=bool)
        for k in range(self.action_dim):
            if np.all(agent_a[:, -1] == 1):
                break

            cur_state[:, 256:] = cur_history
            # print(state.device)
            s = torch.from_numpy(cur_state).to(state.device, state.dtype)
            # print(s.device)
            acts = self.select_action(s, old_acts)
            acts = acts.cpu().numpy()
            # print(acts)

            kept_acts = acts[kept]
            agent_a[kept, kept_acts] = 1
            old_acts[kept, kept_acts] = True

            kept = kept & (acts != self.action_dim - 1)

            # print(kept, kept_acts)
            cur_history[kept, acts[kept]] = 1
        # Sanity check
        assert np.all(agent_a[:, -1] == 1), "Not all action is time_pass:\n" + str(
            agent_a[:, -1]
        )
        return agent_a


class Value(nn.Module):
    """Value (Value) Model."""

    def __init__(self, state_size, hidden_size=64):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
