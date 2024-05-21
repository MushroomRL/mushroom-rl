from copy import deepcopy

import torch
import torch.nn as nn

from mushroom_rl.algorithms.value.dqn import DQN
from mushroom_rl.approximators.parametric import NumpyTorchApproximator


class DuelingNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, features_network, n_features,
                 avg_advantage, **kwargs):
        super().__init__()

        self._avg_advantage = avg_advantage

        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,),
                                     n_features=n_features, **kwargs)

        self._A = nn.Linear(n_features, self._n_output)
        self._V = nn.Linear(n_features, 1)

        nn.init.xavier_uniform_(self._A.weight,
                                gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self._V.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
        features = self._phi(state)

        advantage = self._A(features)
        value = self._V(features)

        q = value + advantage
        if self._avg_advantage:
            q -= advantage.mean(1).reshape(-1, 1)
        else:
            q -= advantage.max(1).reshape(-1, 1)

        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))

            return q_acted


class DuelingDQN(DQN):
    """
    Dueling DQN algorithm.
    "Dueling Network Architectures for Deep Reinforcement Learning".
    Wang Z. et al.. 2016.

    """
    def __init__(self, mdp_info, policy, approximator_params,
                 avg_advantage=True, **params):
        """
        Constructor.

        """
        features_network = approximator_params['network']
        params['approximator_params'] = deepcopy(approximator_params)
        params['approximator_params']['network'] = DuelingNetwork
        params['approximator_params']['features_network'] = features_network
        params['approximator_params']['avg_advantage'] = avg_advantage
        params['approximator_params']['output_dim'] = (mdp_info.action_space.n,)

        super().__init__(mdp_info, policy, NumpyTorchApproximator, **params)
