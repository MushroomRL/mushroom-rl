import torch
import torch.nn as nn
import torch.nn.functional as F

from mushroom_rl.approximators.parametric.networks.noisy_network import NoisyNetwork
from mushroom_rl.utils.torch_utils import TorchUtils

eps = torch.finfo(torch.float32).eps


class RainbowNetwork(nn.Module):
    """
    Network for Rainbow, combining a distributional categorical head (``n_atoms`` over ``[v_min, v_max]``) with
    a dueling architecture built from noisy linear layers.

    """
    def __init__(self, input_shape, output_shape, features_network, n_atoms,
                 v_min, v_max, n_features, sigma_coeff, **kwargs):
        """
        Constructor.

        Args:
            input_shape (tuple): shape of the input (the state);
            output_shape (tuple): shape of the output (the number of actions);
            features_network (nn.Module): the network used to compute the features;
            n_atoms (int): number of atoms of the support of the value distribution;
            v_min (float): minimum value of the support;
            v_max (float): maximum value of the support;
            n_features (int): number of features extracted by the features network;
            sigma_coeff (float): scaling coefficient for the initial noise standard deviation;
            **kwargs: parameters forwarded to the features network.

        """
        super().__init__()

        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,),
                                     n_features=n_features, **kwargs)
        self._n_atoms = n_atoms
        self._v_min = v_min
        self._v_max = v_max

        delta = (self._v_max - self._v_min) / (self._n_atoms - 1)
        self._a_values = torch.arange(self._v_min, self._v_max + eps, delta, device=TorchUtils.get_device())

        self._pv = NoisyNetwork.NoisyLinear(n_features, n_atoms, sigma_coeff)
        self._pa = nn.ModuleList([NoisyNetwork.NoisyLinear(n_features, n_atoms, sigma_coeff)
                                  for _ in range(self._n_output)])

    def forward(self, state, action=None, get_distribution=False, **kwargs):
        features = self._phi(state)

        a_pv = self._pv(features)
        a_pa = [self._pa[i](features) for i in range(self._n_output)]
        a_pa = torch.stack(a_pa, dim=1)
        a_pv = a_pv.unsqueeze(1).repeat(1, self._n_output, 1)
        mean_a_pa = a_pa.mean(1, keepdim=True).repeat(1, self._n_output, 1)
        softmax = F.softmax(a_pv + a_pa - mean_a_pa, dim=-1)

        if not get_distribution:
            q = torch.empty(softmax.shape[:-1])
            for i in range(softmax.shape[0]):
                q[i] = softmax[i] @ self._a_values

            if action is not None:
                return torch.squeeze(q.gather(1, action))
            else:
                return q
        else:
            if action is not None:
                action = torch.unsqueeze(
                    action.long(), 2).repeat(1, 1, self._n_atoms)

                return torch.squeeze(softmax.gather(1, action))
            else:
                return softmax
