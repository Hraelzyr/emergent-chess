import torch.nn as nn


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(192, 1)

    def forward(self, x):
        return self.fc1(x)