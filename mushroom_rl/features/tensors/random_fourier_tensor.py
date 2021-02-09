import torch
import torch.nn as nn

from mushroom_rl.features.tensors import ConstantTensor
from mushroom_rl.utils.torch import to_float_tensor

import numpy as np


class RandomFourierBasis(nn.Module):
    r"""
    Class implementing Random Fourier basis functions. The value of the feature
    is computed using the formula:

    .. math::
        \sin{\dfrac{PX}{\nu}+\varphi}

    where X is the input, m is the vector of the minumum input values (for each
    dimensions) , \Delta is the vector of maximum

    This features have been presented in:

    "Towards generalization and simplicity in continuous control". Rajeswaran A. et Al..
    2017.

    """
    def __init__(self, P, phi, nu, use_cuda):
        r"""
        Constructor.

        Args:
            P (np.ndarray): weights matrix, every weight should be drawn from a normal distribution;
            phi (np.ndarray): bias vector, every weight should be drawn from a uniform distribution in the interval
                [-\pi, \pi);
             values of the input variables, i.e. delta = high - low;
            nu (float):  bandwidth parameter, it should be chosen approximately as the average pairwise distances
                between different observation vectors;
            use_cuda (bool): whether to use cuda for the computation or not.

        """
        self._P = to_float_tensor(P, use_cuda)
        self._phi = to_float_tensor(phi, use_cuda)
        self._nu = nu

        self._use_cuda = use_cuda

    def forward(self, x):
        if self._use_cuda:
            x = x.cuda()
        return torch.sin(x @ self._P / self._nu + self._phi)

    def __str__(self):
        return str(self._P) + ' ' + str(self._phi)

    @staticmethod
    def generate(nu, n_output, input_size, use_cuda=False, use_bias=True):
        """
        Factory method to build random fourier basis. Includes a constant tensor into the output.

        Args:
            nu (float):  bandwidth parameter, it should be chosen approximately as the average pairwise distances
                between different observation vectors.
            n_output (int): number of basis to use;
            input_size (int): size of the input;
            use_cuda (bool): whether to use cuda for the computation or not.

        Returns:
            The list of the generated fourier basis functions.

        """
        if use_bias:
            n_output -= 1

        P = np.random.randn(input_size, n_output)
        phi = np.random.uniform(-np.pi, np.pi, n_output)

        tensor_list = [RandomFourierBasis(P, phi, nu, use_cuda)]

        if use_bias:
            tensor_list.append(ConstantTensor())

        return tensor_list

    @property
    def size(self):
        return self._phi.shape[0]
