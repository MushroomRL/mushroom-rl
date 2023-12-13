import torch
import torch.nn as nn

from mushroom_rl.features.tensors import ConstantTensor
from mushroom_rl.utils.torch import TorchUtils

import numpy as np


class RandomFourierBasis(nn.Module):
    r"""
    Class implementing Random Fourier basis functions. The value of the feature is computed using the formula:

    .. math::
        \sin{\dfrac{PX}{\nu}+\varphi}


    where :math:`X` is the input, :math:`P` is a random weights matrix, :math:`\nu` is the bandwidth parameter and
    :math:`\varphi` is a bias vector.

    These features have been presented in:

    "Towards generalization and simplicity in continuous control". Rajeswaran A. et Al.. 2017.

    """
    def __init__(self, P, phi, nu):
        """
        Constructor.

        Args:
            P (np.ndarray): weights matrix, every weight should be drawn from a normal distribution;
            phi (np.ndarray): bias vector, every weight should be drawn from a uniform distribution in the interval
                :math: `[-\pi, \pi)`;
            nu (float):  bandwidth parameter, it should be chosen approximately as the average pairwise distances
                between different observation vectors.

        """
        self._P = TorchUtils.to_float_tensor(P)
        self._phi = TorchUtils.to_float_tensor(phi)
        self._nu = nu

        super().__init__()

    def forward(self, x):
        return torch.sin(x @ self._P / self._nu + self._phi)

    def __str__(self):
        return str(self._P) + ' ' + str(self._phi)

    @staticmethod
    def generate(nu, n_output, input_size,  use_bias=True):
        """
        Factory method to build random fourier basis. Includes a constant tensor into the output.

        Args:
            nu (float):  bandwidth parameter, it should be chosen approximately as the average pairwise distances
                between different observation vectors.
            n_output (int): number of basis to use;
            input_size (int): size of the input.

        Returns:
            The list of the generated fourier basis functions.

        """
        if use_bias:
            n_output -= 1

        P = np.random.randn(input_size, n_output)
        phi = np.random.uniform(-np.pi, np.pi, n_output)

        tensor_list = [RandomFourierBasis(P, phi, nu)]

        if use_bias:
            tensor_list.append(ConstantTensor())

        return tensor_list

    @property
    def size(self):
        return self._phi.shape[0]
