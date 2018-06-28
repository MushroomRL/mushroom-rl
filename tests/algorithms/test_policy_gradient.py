import numpy as np

from mushroom.algorithms.policy_search import *
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import LinearApproximator
from mushroom.core import Core
from mushroom.environments.lqr import LQR
from mushroom.policy.gaussian_policy import StateStdGaussianPolicy
from mushroom.utils.parameters import AdaptiveParameter

mdp = LQR.generate(dimensions=1)

approximator_params = dict(input_dim=mdp.info.observation_space.shape)
approximator = Regressor(LinearApproximator,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=mdp.info.action_space.shape,
                         params=approximator_params)

sigma = Regressor(LinearApproximator,
                  input_shape=mdp.info.observation_space.shape,
                  output_shape=mdp.info.action_space.shape,
                  params=approximator_params)

sigma_weights = 2 * np.ones(sigma.weights_size)
sigma.set_weights(sigma_weights)

policy = StateStdGaussianPolicy(approximator, sigma)

# Agent
learning_rate = AdaptiveParameter(value=.01)
algorithm_params = dict(learning_rate=learning_rate)

agent_test = REINFORCE(policy, mdp.info, **algorithm_params)
core = Core(agent_test, mdp)

s = np.arange(10)
a = np.arange(10)
r = np.arange(10)
ss = s + 5
ab = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0])
last = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0])

dataset = list()
for i in range(s.size):
    dataset.append([np.array([s[i]]), np.array([a[i]]), r[i],
                    np.array([ss[i]]), np.array([ab[i]]), np.array([last[i]])])


def test_REINFORCE():
    agent = REINFORCE(policy, mdp.info, **algorithm_params)
    agent.fit(dataset)

    w = np.array([-.07071067, 2.07071068])

    assert np.allclose(w, agent.policy.get_weights())


def test_GPOMDP():
    agent = GPOMDP(policy, mdp.info, **algorithm_params)
    agent.fit(dataset)

    w = np.array([-.12837306, 2.15241166])

    assert np.allclose(w, agent.policy.get_weights())


def test_eNAC():
    agent = eNAC(policy, mdp.info, **algorithm_params)
    agent.fit(dataset)

    w = np.array([-.0305895, 2.13147422])

    assert np.allclose(w, agent.policy.get_weights())
