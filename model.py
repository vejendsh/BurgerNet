import torch.nn as nn
import torch
from data_generation import num_grid_points


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(torch.floor(torch.tensor(num_grid_points)).item() + 1, 264),
            nn.LeakyReLU(negative_slope=2),
            nn.Linear(264, 328),
            nn.LeakyReLU(negative_slope=2),
            nn.Linear(328, 464),
            nn.LeakyReLU(negative_slope=2),
            nn.Linear(464, 328),
            nn.LeakyReLU(negative_slope=2),
            nn.Linear(328, 264),
            nn.LeakyReLU(negative_slope=2),
            nn.Linear(264, torch.floor(torch.tensor(num_grid_points)).item() + 1),
            nn.LeakyReLU(negative_slope=2)

        )

    def forward(self, x):
        pred = self.linear_relu_stack(x)
        return pred
