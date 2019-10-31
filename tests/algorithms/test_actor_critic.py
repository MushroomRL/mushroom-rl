import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mushroom.algorithms.actor_critic import *
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import LinearApproximator
from mushroom.core import Core
from mushroom.environments import InvertedPendulum
from mushroom.features import Features
from mushroom.features.tiles import Tiles
from mushroom.policy import GaussianTorchPolicy
from mushroom.policy.gaussian_policy import GaussianPolicy
from mushroom.utils.parameters import Parameter


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Network, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, **kwargs):
        features1 = torch.tanh(self._h1(torch.squeeze(state, 1).float()))
        features2 = torch.tanh(self._h2(features1))
        a = self._h3(features2)

        return a


torch.manual_seed(1)

n_steps = 5000
mdp = InvertedPendulum(horizon=n_steps)

# Agent
n_tilings = 10
alpha_theta = Parameter(5e-3 / n_tilings)
alpha_omega = Parameter(0.5 / n_tilings)
alpha_v = Parameter(0.5 / n_tilings)
tilings = Tiles.generate(n_tilings, [10, 10],
                         mdp.info.observation_space.low,
                         mdp.info.observation_space.high + 1e-3)

phi = Features(tilings=tilings)

input_shape = (phi.size,)

mu = Regressor(LinearApproximator, input_shape=input_shape,
               output_shape=mdp.info.action_space.shape)

sigma = 1e-1 * np.eye(1)
policy = GaussianPolicy(mu, sigma)

agent_test = COPDAC_Q(policy, mu, mdp.info,
                      alpha_theta, alpha_omega, alpha_v,
                      value_function_features=phi,
                      policy_features=phi)

core = Core(agent_test, mdp)

s = np.array([np.arange(10), np.arange(10)]).T
a = np.arange(10)
r = np.arange(10)
ss = s + 5
ab = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0])
last = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0])

dataset = list()
for i in range(s.shape[0]):
    dataset.append([s[i], np.array([a[i]]), r[i], ss[i], np.array([ab[i]]),
                    np.array([last[i]])])


def test_copdac_q():
    agent = COPDAC_Q(policy, mu, mdp.info, alpha_theta, alpha_omega, alpha_v,
                     value_function_features=phi, policy_features=phi)
    agent.fit(dataset)

    w_1 = .05
    w_2 = .1

    w = agent._V.get_weights()

    assert np.allclose(w_1, w[66])
    assert np.allclose(w_2, w[78])


def test_sac():
    tilings_v = tilings + Tiles.generate(1, [1, 1],
                                         mdp.info.observation_space.low,
                                         mdp.info.observation_space.high + 1e-3)
    psi = Features(tilings=tilings_v)
    agent = StochasticAC(policy, mdp.info, alpha_theta, alpha_v, lambda_par=.5,
                         value_function_features=psi, policy_features=phi)
    agent.fit(dataset)

    w_1 = .09370429
    w_2 = .28141735

    w = agent._V.get_weights()

    assert np.allclose(w_1, w[65])
    assert np.allclose(w_2, w[78])


def test_sac_avg():
    alpha_r = Parameter(.0001)
    tilings_v = tilings + Tiles.generate(1, [1, 1],
                                         mdp.info.observation_space.low,
                                         mdp.info.observation_space.high + 1e-3)
    psi = Features(tilings=tilings_v)
    agent = StochasticAC_AVG(policy, mdp.info, alpha_theta, alpha_v, alpha_r,
                             lambda_par=.5, value_function_features=psi,
                             policy_features=phi)
    agent.fit(dataset)

    w_1 = .09645764
    w_2 = .28583057

    w = agent._V.get_weights()

    assert np.allclose(w_1, w[65])
    assert np.allclose(w_2, w[78])


def test_a2c():
    policy_params = dict(
        std_0=1.,
        n_features=64,
        use_cuda=False
    )

    alg_params = dict(actor_optimizer={'class': optim.RMSprop,
                                       'params': {'lr': 7e-4,
                                                  'eps': 3e-3}},
                      max_grad_norm=0.5,
                      ent_coeff=0.01)

    critic_params = dict(network=Network,
                         optimizer={'class': optim.RMSprop,
                                    'params': {'lr': 7e-4,
                                               'eps': 1e-5}},
                         loss=F.mse_loss,
                         n_features=64,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    policy = GaussianTorchPolicy(Network,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    agent = A2C(mdp.info, policy, critic_params, **alg_params)

    agent.fit(dataset)

    w_1 = -0.20738243
    w_2 = -0.3641879

    w = agent._V.get_weights()

    assert np.allclose(w[8], w_1)
    assert np.allclose(w[19], w_2)
