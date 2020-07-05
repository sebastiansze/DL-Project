from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add
import collections


class DQNAgents(object):
    def __init__(self, params):
        self._map_size = [5, 5]

        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']        
        self.epsilon = 1
        self.actual = []
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        # old_states, actions, rewards, new_states, conditions
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.model = self.network()

    def network(self):
        model = Sequential()
        model.add(Dense(self.first_layer, activation='sigmoid', input_dim=4))
        model.add(Dense(self.second_layer, activation='sigmoid'))
        model.add(Dense(self.third_layer, activation='sigmoid'))
        model.add(Dense(4, activation='sigmoid'))  # softmax  # TODO: Warum?
        opt = SGD(self.learning_rate)  # Adam
        model.compile(loss='mse', optimizer=opt)

        if self.load_weights:
            model.load_weights(self.weights)
        return model

    def prepare_model_input(self, x):
        obstacle_maps = x[:, 0]
        own_current_positions_coordinates = np.argwhere(x[:, 1])[:, 1:3] / self._map_size  # normalize coordinates
        own_aim_positions_coordinates = np.argwhere(x[:, 2])[:, 1:3] / self._map_size  # normalize coordinates
        others_current_positions_maps = x[:, 3]

        output = np.concatenate([own_current_positions_coordinates, own_aim_positions_coordinates], axis=1)
        return output

    def predict(self, x):
        return self.model.predict(self.prepare_model_input(x))

    def remember(self, old_states, actions, rewards, new_states, conditions):
        for o_state, action, reward, n_state, condition in zip(old_states, actions, rewards, new_states, conditions):
            self.memory.append((o_state, action, reward, n_state, condition))

    def replay_new(self, memory, batch_size):
        if len(memory) > batch_size:
            mini_batch = random.sample(memory, batch_size)
        else:
            mini_batch = memory

        old_states = []
        actions = []
        rewards = []
        new_states = []
        conditions = []
        for o_state, action, reward, n_state, condition in mini_batch:
            old_states.append(o_state)
            actions.append(action)
            rewards.append(reward)
            new_states.append(n_state)
            conditions.append(condition)

        self.train_short_memory(np.array(old_states),
                                np.array(actions),
                                np.array(rewards),
                                np.array(new_states),
                                np.array(conditions))

    def train_short_memory(self, old_states, actions, rewards, new_states, conditions):
        """

        :param old_states: shape: [agent_count, 4, size_x, size_y]
        :param actions: shape: [agent_count, 5]
        :param rewards: shape: [agent_count, 5]
        :param new_states: shape: [agent_count, 4, size_x, size_y]
        :param conditions: shape: [agent_count]
        :return:
        """

        targets = np.where(np.isin(conditions, ['a', 's', '3']),
                           rewards,
                           rewards + self.gamma * np.amax(self.predict(new_states)))
        target_f = self.predict(old_states)
        target_f[:, np.argmax(actions)] = targets
        # print('  - {}'.format(target_f))

        self.model.fit(self.prepare_model_input(old_states), target_f, epochs=1, verbose=0)
