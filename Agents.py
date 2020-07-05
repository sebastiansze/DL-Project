import numpy as np
import torch
from ConvolutionalDQN import ConvolutionalDQN
from map import print_layers


class Agents:
    def __init__(self):
        self.gamma = 0.9

        self.net = ConvolutionalDQN()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.07)
        self.loss_function = torch.nn.MSELoss()

        if torch.cuda.is_available():
            print('Cuda available')
            model = self.net.cuda()
            criterion = self.loss_function.cuda()

    def predict(self, x):
        # clearing the Gradients of the model parameters
        self.optimizer.zero_grad()

        return self.net.forward(x).data.numpy()

    def train(self, y_pred, y_true):
        self.net.train()
        tr_loss = 0

        y_pred = torch.from_numpy(y_pred.astype('float32'))
        y_pred.requires_grad = True

        y_true = torch.from_numpy(y_true)
        y_true.requires_grad = True

        # converting the data into GPU format
        if torch.cuda.is_available():
            y_pred = y_pred.cuda()
            y_true = y_true.cuda()

        # computing the training and validation loss
        loss = self.loss_function(y_pred, y_true)

        # computing the updated weights of all the model parameters
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory, batch_size):
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

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
        self.train(old_states, target_f)
