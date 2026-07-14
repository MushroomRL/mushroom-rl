import torch
import torch.nn as nn
import torch.nn.functional as F

from mushroom_rl.utils.torch_utils import TorchUtils

eps = torch.finfo(torch.float32).eps


class CategoricalNetwork(nn.Module):
    """
    Distributional network for Categorical DQN (C51), modeling the value distribution of each action as a
    categorical distribution over a fixed support of ``n_atoms`` between ``v_min`` and ``v_max``.

    """
    def __init__(self, input_shape, output_shape, features_network, n_atoms,
                 v_min, v_max, n_features, **kwargs):
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

        self._p = nn.ModuleList(
            [nn.Linear(n_features, n_atoms) for _ in range(self._n_output)])

        for i in range(self._n_output):
            nn.init.xavier_uniform_(self._p[i].weight,
                                    gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None, get_distribution=False, **kwargs):
        features = self._phi(state)

        a_p = [F.softmax(self._p[i](features), -1) for i in range(self._n_output)]
        a_p = torch.stack(a_p, dim=1)

        if not get_distribution:
            q = torch.empty(a_p.shape[:-1])
            for i in range(a_p.shape[0]):
                q[i] = a_p[i] @ self._a_values

            if action is not None:
                return torch.squeeze(q.gather(1, action))
            else:
                return q
        else:
            if action is not None:
                action = torch.unsqueeze(
                    action.long(), 2).repeat(1, 1, self._n_atoms)

                return torch.squeeze(a_p.gather(1, action))
            else:
                return a_p
