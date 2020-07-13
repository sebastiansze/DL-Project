from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from map import print_layers

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, padding=1)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(3, 3, 5, padding=2)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(3, 1, 3, padding=1)
        self.fc1 = nn.Linear(9, 5)
        self._weights = None

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        output = self.fc1(x.flatten(start_dim=1))
        # output = F.softmax(output, dim=1)
        return output

    def generate_agents(self, agent_count):
        return [Agent(weights=self._weights) for _ in range(agent_count)]

    def backward(self, predicted, truth):
        # TODO
        pass


class Agent:
    def __init__(self, network: nn.Module):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = network.to(self.device)

    def predict(self, input_map):
        """
        Do forward pass
        :return:
        """

        obstacle_map = input_map[0]
        own_current_position_coordinates = np.argwhere(input_map[1])[0]
        own_aim_position_coordinates = np.argwhere(input_map[2])[0]
        others_current_positions_map = input_map[3]

        print_layers(obstacle_map)
        print(own_current_position_coordinates)
        print(own_aim_position_coordinates)
        print_layers(others_current_positions_map)
        print()
        print('Hier steht ein "input()". Deshalb gehts nicht weiter...')
        input()

        # TODO
        y = self.net(torch.from_numpy(input_map))
        return y
