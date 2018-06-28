import numpy as np

from mushroom.algorithms.actor_critic import *
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import LinearApproximator
from mushroom.core import Core
from mushroom.environments import InvertedPendulum
from mushroom.features import Features
from mushroom.features.tiles import Tiles
from mushroom.policy.gaussian_policy import GaussianPolicy
from mushroom.utils.parameters import Parameter


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


def test_copdac_q():
    agent = COPDAC_Q(policy, mu, mdp.info, alpha_theta, alpha_omega, alpha_v,
                     value_function_features=phi, policy_features=phi)
    agent.fit(dataset)

    np.save('w.npy', agent.policy.get_weights())

    w = np.array([-.07071067, 2.07071068])

    assert np.allclose(w, agent.policy.get_weights())


def test_sac():
    pass


def test_sac_avg():
    pass
