import numpy as np
import torch as T

from ReplayBuffer import ReplayBuffer
from DuelingLinearDeepQNetwork import DuelingLinearDeepQNetwork


def get_one_hot_vectors(indices, length=5):
    return np.array([np.eye(1, length, int(x))[0] for x in indices])


class Agent:
    def __init__(self, params):
        self.gamma = params['gamma']
        self.epsilon = params['epsilon']
        self.lr = params['learning_rate']
        self.n_actions = 5 if params['with_stay_action'] else 4
        self.input_dims = [params['map_size_max'], params['map_size_max']]
        self.batch_size = params['batch_size']
        self.eps_min = params['eps_min']
        self.eps_dec = params['eps_dec']
        self.replace_target_cnt = params['replace']
        self.chkpt_dir = params['weights_path']
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        print(params['memory_size'])
        print(self.input_dims)
        print(self.n_actions)
        self.memory = ReplayBuffer(params['memory_size'], self.input_dims)

        self.q_eval = DuelingLinearDeepQNetwork(self.lr, self.n_actions,
                                                input_dims=self.input_dims,
                                                name='model',
                                                chkpt_dir=self.chkpt_dir)

        self.q_next = DuelingLinearDeepQNetwork(self.lr, self.n_actions,
                                                input_dims=self.input_dims,
                                                name='model_next',
                                                chkpt_dir=self.chkpt_dir)

    def get_states_from_observations(self, x):
        obstacle_maps = x[:, 0]
        own_current_positions_coordinates = np.argwhere(x[:, 1])[:, 1:3] / self.input_dims[0]  # normalize coordinates
        own_aim_positions_coordinates = np.argwhere(x[:, 2])[:, 1:3] / self.input_dims[1]  # normalize coordinates
        others_current_positions_maps = x[:, 3]
        # ts = time_steps / self._time_step_max
        states = np.concatenate([own_current_positions_coordinates, own_aim_positions_coordinates], axis=1)
        return T.tensor(states, dtype=T.float).to(self.q_eval.device)

    def choose_actions(self, observations):
        agent_count = observations.shape[0]

        states = self.get_states_from_observations(observations)
        _, predictions = self.q_eval.forward(states)

        predictions_ext = np.zeros((agent_count, 5))
        predictions_ext[:, 5-self.n_actions:5] = predictions.data.numpy()

        random_choise = np.random.random(size=(agent_count, 1)) > self.epsilon
        actions = np.where(random_choise,  # Choose randomly between ...
                           get_one_hot_vectors(np.argmax(predictions_ext, axis=1)),  # predicted actions.
                           get_one_hot_vectors(np.random.randint(1, 5, size=agent_count)))  # random actions and ...

        return actions

    def store_transition(self, old_obs, actions, rewards, new_obs, done_list):
        old_states = self.get_states_from_observations(old_obs)
        new_states = self.get_states_from_observations(new_obs)
        for o_state, action, reward, n_state, done in zip(old_states, actions, rewards, new_states, done_list):
            self.memory.store_transition(o_state, np.argmax(action), reward, n_state, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        if self.n_actions == 4:
            action -= 1

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
