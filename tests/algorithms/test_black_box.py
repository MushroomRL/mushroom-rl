import numpy as np

from mushroom.algorithms.policy_search import PGPE, REPS, RWR
from mushroom.approximators import Regressor
from mushroom.core import Core
from mushroom.approximators.parametric import LinearApproximator
from mushroom.distributions import GaussianDiagonalDistribution
from mushroom.environments import ShipSteering
from mushroom.features import Features
from mushroom.features.tiles import Tiles
from mushroom.policy import DeterministicPolicy
from mushroom.utils.parameters import AdaptiveParameter

mdp = ShipSteering()

high = [150, 150, np.pi]
low = [0, 0, -np.pi]
n_tiles = [5, 5, 6]
low = np.array(low, dtype=np.float)
high = np.array(high, dtype=np.float)
n_tilings = 1

tilings = Tiles.generate(n_tilings=n_tilings, n_tiles=n_tiles, low=low,
                         high=high)

phi = Features(tilings=tilings)
input_shape = (phi.size,)

approximator = Regressor(LinearApproximator, input_shape=input_shape,
                         output_shape=mdp.info.action_space.shape)

policy = DeterministicPolicy(approximator)

mu = np.zeros(policy.weights_size)
sigma = 4e-1 * np.ones(policy.weights_size)
distribution_test = GaussianDiagonalDistribution(mu, sigma)
agent_test = RWR(distribution_test, policy, mdp.info, beta=1.)
core = Core(agent_test, mdp)

s = np.arange(10)
a = np.arange(10)
r = np.arange(10)
ss = s + 5
ab = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
last = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

dataset = list()
for i in range(s.size):
    dataset.append([np.array([s[i]]), np.array([a[i]]), r[i],
                    np.array([ss[i]]), ab[i], last[i]])

np.random.seed(88)


def test_RWR():
    distribution = GaussianDiagonalDistribution(mu, sigma)
    agent = RWR(distribution, policy, mdp.info, beta=1., features=phi)

    agent.episode_start()

    agent.fit(dataset)

    w_1 = 4.24574375e-1
    w_2 = -1.10809513e-1

    w = agent.policy.get_weights()

    assert np.allclose(w_1, w[10])
    assert np.allclose(w_2, w[18])


def test_REPS():
    distribution = GaussianDiagonalDistribution(mu, sigma)
    agent = REPS(distribution, policy, mdp.info, eps=.7, features=phi)

    agent.episode_start()

    agent.fit(dataset)

    w_1 = .76179551
    w_2 = .08787432

    w = agent.policy.get_weights()

    assert np.allclose(w_1, w[10])
    assert np.allclose(w_2, w[18])


def test_PGPE():
    distribution = GaussianDiagonalDistribution(mu, sigma)
    agent = PGPE(distribution, policy, mdp.info,
                 learning_rate=AdaptiveParameter(1.5), features=phi)

    agent.episode_start()

    agent.fit(dataset)

    w_1 = .54454343
    w_2 = .5449792

    w = agent.policy.get_weights()

    assert np.allclose(w_1, w[10])
    assert np.allclose(w_2, w[18])
