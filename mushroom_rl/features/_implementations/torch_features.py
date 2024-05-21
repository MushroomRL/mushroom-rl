import numpy as np
import torch

from .features_implementation import FeaturesImplementation
from mushroom_rl.utils.torch import TorchUtils


class TorchFeatures(FeaturesImplementation):
    def __init__(self, tensor_list):
        self._phi = tensor_list

    def __call__(self, *args):
        x = self._concatenate(args)

        x = TorchUtils.to_float_tensor(np.atleast_2d(x))

        y_list = [self._phi[i].forward(x) for i in range(len(self._phi))]
        y = torch.cat(y_list, 1).squeeze()

        y = y.detach().cpu().numpy()

        if y.shape[0] == 1:
            return y[0]
        else:
            return y

    @property
    def size(self):
        return np.sum([phi.size for phi in self._phi])
