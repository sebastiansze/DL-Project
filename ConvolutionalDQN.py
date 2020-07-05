import numpy as np
import torch


class ConvolutionalDQN(torch.nn.Module):
    def __init__(self):
        super(ConvolutionalDQN, self).__init__()
        map_size = [25, 25]
        self.model = torch.nn.Sequential(
            # Defining a 3D convolution layer
            # torch.nn.Conv3d(4, 4, kernel_size=5, stride=1, padding=2),
            # torch.nn.BatchNorm3d(4),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            # torch.nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            # torch.nn.BatchNorm2d(4),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(1, -1),
            torch.nn.Linear(4 * map_size[0] * map_size[1], 2500),
            torch.nn.ReLU(),
            # torch.nn.Linear(2500, 2500),
            # torch.nn.ReLU(),
            torch.nn.Linear(2500, 5),
        )

    def forward(self, x):
        # obstacle_map = x[0]
        # own_current_position_coordinates = np.argwhere(x[1])[0] / self._map_size  # normalize coordinates
        # own_aim_position_coordinates = np.argwhere(x[2])[0] / self._map_size  # normalize coordinates
        # others_current_positions_map = x[3]

        print(x.shape)
        x = torch.from_numpy(x.astype('float32'))
        y_pred = self.model(x)
        return y_pred
