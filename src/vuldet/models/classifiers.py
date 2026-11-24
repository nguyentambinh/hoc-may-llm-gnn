# Nguyễn Tâm Bình - 2324801030003
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden=(512, 128), dropout=0.3):
        super().__init__()
        layers, previous = [], input_dim
        for size in hidden:
            layers += [nn.Linear(previous, size), nn.ReLU(), nn.Dropout(dropout)]
            previous = size
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
