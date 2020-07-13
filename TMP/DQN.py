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
        self._map_size_max = [params['map_size_max'], params['map_size_max']]
        self._max_possible_dist = np.sqrt(np.sum(np.square(self._map_size_max)))
        self._time_step_max = params['time_step_max']

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
        model.add(Dense(self.first_layer, activation='relu', input_dim=5))
        model.add(Dense(self.second_layer, activation='relu'))
        model.add(Dense(self.third_layer, activation='relu'))
        model.add(Dense(4, activation='softmax'))  # softmax  # TODO: Warum?
        opt = SGD(self.learning_rate)  # Adam
        model.compile(loss='mse', optimizer=opt)

        if self.load_weights:
            model.load_weights(self.weights)
        return model

    def get_reward(self, conditions, distances_to_aims):
        condition_list = [conditions == c for c in ['a', 's', '3', 't']]
        choice_list = [+2 * self._max_possible_dist,  # aim
                       -2 * self._max_possible_dist,  # self inflicted accident
                       -1 * self._max_possible_dist,  # third-party fault accident
                       -1 * self._max_possible_dist]  # time out
        # dist_func = 1 / (2 * distances_to_aims + 1) + 0.5
        # return np.select(condition_list, [1, 0, 0.2], dist_func)
        dist_func = self._max_possible_dist - distances_to_aims
        return np.select(condition_list, choice_list, dist_func)

    def prepare_model_input(self, x, time_steps):
        obstacle_maps = x[:, 0]
        own_current_positions_coordinates = np.argwhere(x[:, 1])[:, 1:3] / self._map_size_max  # normalize coordinates
        own_aim_positions_coordinates = np.argwhere(x[:, 2])[:, 1:3] / self._map_size_max  # normalize coordinates
        others_current_positions_maps = x[:, 3]
        ts = time_steps / self._time_step_max
        output = np.concatenate([own_current_positions_coordinates, own_aim_positions_coordinates, ts], axis=1)
        return output

    def predict(self, x, time_step):
        return self.model.predict(self.prepare_model_input(x, time_step))

    def remember(self, old_states, time_steps, actions, rewards, new_states, conditions):
        for o_state, ts, action, reward, n_state, condition in zip(old_states, time_steps, actions, rewards,
                                                                   new_states, conditions):
            self.memory.append((o_state, ts, action, reward, n_state, condition))

    def replay_new(self, memory, batch_size):
        if len(memory) > batch_size:
            mini_batch = random.sample(memory, batch_size)
        else:
            mini_batch = memory

        old_states = []
        time_steps = []
        actions = []
        rewards = []
        new_states = []
        conditions = []
        for o_state, ts, action, reward, n_state, condition in mini_batch:
            old_states.append(o_state)
            time_steps.append(ts)
            actions.append(action)
            rewards.append(reward)
            new_states.append(n_state)
            conditions.append(condition)

        self.train_short_memory(np.array(old_states),
                                np.array(time_steps),
                                np.array(actions),
                                np.array(rewards),
                                np.array(new_states),
                                np.array(conditions))

    def train_short_memory(self, old_states, time_steps, actions, rewards, new_states, conditions):
        """

        :param old_states: shape: [agent_count, 4, size_x, size_y]
        :param time_steps:
        :param actions: shape: [agent_count, 5]
        :param rewards: shape: [agent_count, 5]
        :param new_states: shape: [agent_count, 4, size_x, size_y]
        :param conditions: shape: [agent_count]
        :return:
        """

        targets = np.where(np.isin(conditions, ['a', 's', '3', 't']),
                           rewards,
                           rewards + self.gamma * np.amax(self.predict(new_states, time_steps + 1)))
        target_f = self.predict(old_states, time_steps)
        target_f[np.arange(target_f.shape[0]), np.argmax(actions, axis=1)] = targets
        # print('  - {}'.format(target_f))

        self.model.fit(self.prepare_model_input(old_states, time_steps), target_f, epochs=1, verbose=0)
