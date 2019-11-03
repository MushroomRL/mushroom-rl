import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mushroom.algorithms.actor_critic import PPO, TRPO
from mushroom.core import Core
from mushroom.environments import Gym
from mushroom.policy import GaussianTorchPolicy


def learn(alg, alg_params):
    class Network(nn.Module):
        def __init__(self, input_shape, output_shape, n_features, **kwargs):
            super(Network, self).__init__()

            n_input = input_shape[-1]
            n_output = output_shape[0]

            self._h = nn.Linear(n_input, n_output)

            nn.init.xavier_uniform_(self._h.weight,
                                    gain=nn.init.calculate_gain('relu'))

        def forward(self, state, **kwargs):
            return F.relu(self._h(torch.squeeze(state, 1).float()))

    mdp = Gym('Pendulum-v0', 200, .99)
    mdp.seed(1)

    critic_params = dict(network=Network,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=4,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    policy_params = dict(std_0=1., n_features=4, use_cuda=False)

    policy = GaussianTorchPolicy(Network,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    agent = alg(mdp.info, policy, critic_params, **alg_params)

    core = Core(agent, mdp)

    core.learn(n_episodes=2, n_episodes_per_fit=1)

    return policy


def test_PPO():
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    params = dict(actor_optimizer={'class': optim.Adam,
                                   'params': {'lr': 3e-4}},
                  n_epochs_policy=4, batch_size=64, eps_ppo=.2, lam=.95,
                  quiet=True)
    policy = learn(PPO, params)
    w = policy.get_weights()
    w_test = np.array([-1.6293062, 1.0408604, -3.5757786e-1, 2.6958251e-1,
                       -8.7002787e-4])

    assert np.allclose(w, w_test)


def test_TRPO():
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    params = dict(ent_coeff=0.0, max_kl=.001, lam=.98, n_epochs_line_search=10,
                  n_epochs_cg=10, cg_damping=1e-2, cg_residual_tol=1e-10,
                  quiet=True)
    policy = learn(TRPO, params)
    w = policy.get_weights()
    w_test = np.array([-1.5759772, 1.0822705, -0.37794656, 0.29728204,
                       -0.0396419])

    assert np.allclose(w, w_test)
