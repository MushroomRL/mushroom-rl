from copy import deepcopy

from mushroom_rl.algorithms.value.dqn import DQN
from mushroom_rl.approximators.parametric import NumpyTorchApproximator
from mushroom_rl.approximators.parametric.networks import DuelingNetwork


class DuelingDQN(DQN):
    """
    Dueling DQN algorithm.
    "Dueling Network Architectures for Deep Reinforcement Learning"
    Wang Z. et al. 2016.

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
