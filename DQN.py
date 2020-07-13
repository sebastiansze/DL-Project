

import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch as T
import imageio


class DuelingLinearDeepQNetwork(nn.Module):
    def __init__(self, ALPHA, n_actions, name, input_dims, chkpt_dir=''):
        super(DuelingLinearDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)

        self.preV = nn.Linear(256, 128)
        self.V = nn.Linear(128, 1)

        self.preA = nn.Linear(256, 128)
        self.A = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_dqn')

    def forward(self, state):
        l1 = F.relu(self.fc1(state))
        l2 = F.relu(self.fc2(l1))
        l3 = F.relu(self.fc3(l2))
        prV = F.relu(self.preV(l3))
        V = self.V(prV)
        prA = F.relu(self.preA(l3))
        A = self.A(prA)

        return V, A

    def save_checkpoint(self):
        # print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        # print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))