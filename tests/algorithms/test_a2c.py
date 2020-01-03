import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from mushroom_rl.core import Core
from mushroom_rl.environments import Gym
from mushroom_rl.algorithms.actor_critic import A2C

from mushroom_rl.policy import GaussianTorchPolicy


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h = nn.Linear(n_input, n_output)

        nn.init.xavier_uniform_(self._h.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, state):
        return F.relu(self._h(torch.squeeze(state, 1).float()))


def test_a2c():
    mdp = Gym(name='Pendulum-v0', horizon=200, gamma=.99)
    mdp.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    policy_params = dict(
        std_0=1.,
        n_features=64,
        use_cuda=False
    )

    critic_params = dict(network=Network,
                         optimizer={'class': optim.RMSprop,
                                    'params': {'lr': 7e-4,
                                               'eps': 1e-5}},
                         loss=F.mse_loss,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    algorithm_params = dict(critic_params=critic_params,
                            actor_optimizer={'class': optim.RMSprop,
                                             'params': {'lr': 7e-4,
                                                        'eps': 3e-3}},
                            max_grad_norm=0.5,
                            ent_coeff=0.01)

    policy = GaussianTorchPolicy(Network,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    agent = A2C(mdp.info, policy, **algorithm_params)

    core = Core(agent, mdp)

    core.learn(n_episodes=10, n_episodes_per_fit=5)

    w = agent.policy.get_weights()
    w_test = np.array([-1.6307759, 1.0356185, -0.34508315, 0.27108294,
                       -0.01047843])

    assert np.allclose(w, w_test)
