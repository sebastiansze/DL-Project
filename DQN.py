import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch as T


class DuelingLinearDeepQNetwork(nn.Module):
    def __init__(self, alpha, n_actions, input_dims,viewReduced=False):
        super(DuelingLinearDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)

        self.preV = nn.Linear(260, 128)
        self.V = nn.Linear(128, 1)

        self.preA = nn.Linear(260, 128)
        self.A = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.viewReduced = viewReduced

    def forward(self, state):
        if self.viewReduced:
            l1 = F.relu(self.fc1(state))
            l2 = F.relu(self.fc2(l1))
            l3 = F.relu(self.fc3(l2))
            prV = F.relu(self.preV(l3))
            V = self.V(prV)
            prA = F.relu(self.preA(l3))
            A = self.A(prA)
        else:
            data = state[:, -4:]
            l1 = F.relu(self.fc1(state))
            l2 = F.relu(self.fc2(l1))
            l3 = F.relu(self.fc3(l2))
            l4 = T.cat((l3, data), dim=1)
            prV = F.relu(self.preV(l4))
            V = self.V(prV)
            prA = F.relu(self.preA(l4))
            A = self.A(prA)

        return V, A

    def save_checkpoint(self, file):
        # print('... saving checkpoint ...')
        T.save(self.state_dict(), file)

    def load_checkpoint(self, file):
        # print('... loading checkpoint ...')
        self.load_state_dict(T.load(file))
