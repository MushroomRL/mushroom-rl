import torch
import torch.nn as nn

from mushroom_rl.utils.torch import TorchUtils


class ConstantTensor(nn.Module):
    """
    Pytorch module to implement a constant function (always one).

    """

    def forward(self, x):
        return torch.ones(x.shape[0], 1).to(TorchUtils.get_device())

    @property
    def size(self):
        return 1
