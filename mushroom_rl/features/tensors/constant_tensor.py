import torch
import torch.nn as nn


class ConstantTensor(nn.Module):
    """
    Pytorch module to implement a constant function (always one).

    """

    def forward(self, x):
        return torch.ones(x.shape[0], 1)

    @property
    def size(self):
        return 1
