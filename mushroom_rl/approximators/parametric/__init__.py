from .linear import LinearApproximator
from .torch_approximator import TorchApproximator, TorchEnsemble
from .numpy_torch_approximator import NumpyTorchApproximator
from .recurrent_torch_approximator import RecurrentTorchApproximator, RecurrentTorchEnsemble
from .cmac import CMAC


__all__ = ['LinearApproximator', 'TorchApproximator', 'TorchEnsemble', 'NumpyTorchApproximator',
           'RecurrentTorchApproximator', 'RecurrentTorchEnsemble', 'CMAC']
