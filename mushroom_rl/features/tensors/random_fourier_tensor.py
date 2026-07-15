import torch
import torch.nn as nn

from mushroom_rl.utils.torch_utils import TorchUtils

from .constant_tensor import ConstantTensor


class RandomFourierTensor(nn.Module):
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
        r"""
        Constructor.

        Args:
            P (Array): weights matrix, every weight should be drawn from a normal distribution;
            phi (Array): bias vector, every weight should be drawn from a uniform distribution in the interval
                :math: `[-\pi, \pi)`;
            nu (float):  bandwidth parameter, it should be chosen approximately as the average pairwise distances
                between different observation vectors.

        """
        super().__init__()

        self.register_buffer('_P', TorchUtils.to_float_tensor(P))
        self.register_buffer('_phi', TorchUtils.to_float_tensor(phi))
        self._nu = nu

    def extra_repr(self):
        return f'n_output={self.size}, nu={self._nu}'

    def forward(self, x):
        return torch.sin(x @ self._P / self._nu + self._phi)

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

        P = torch.randn(input_size, n_output)
        phi = torch.rand(n_output) * 2 * torch.pi - torch.pi

        tensor_list = [RandomFourierTensor(P, phi, nu)]

        if use_bias:
            tensor_list.append(ConstantTensor())

        return tensor_list

    @property
    def size(self):
        return self._phi.shape[0]
