from copy import deepcopy

from mushroom_rl.algorithms.value.dqn import DQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import NoisyNetwork


class NoisyDQN(DQN):
    """
    Noisy DQN algorithm.
    "Noisy networks for exploration"
    Fortunato M. et al. 2018.

    """
    def __init__(self, mdp_info, policy, approximator_params, **params):
        """
        Constructor.

        """
        features_network = approximator_params['network']
        params['approximator_params'] = deepcopy(approximator_params)
        params['approximator_params']['network'] = NoisyNetwork
        params['approximator_params']['features_network'] = features_network

        super().__init__(mdp_info, policy, TorchApproximator, **params)
