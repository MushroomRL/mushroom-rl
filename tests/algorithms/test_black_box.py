import numpy as np
import pytest
import torch
from torch import optim
from functools import partial

from helper.dummies import FakeLogger
from helper.cases import devices
from helper.utils import TestUtils as tu

from mushroom_rl.algorithms.policy_search import PGPE, REPS, ConstrainedREPS, RWR, MORE, ePPO
from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization
from mushroom_rl.core import Core, MultiprocessEnvironment
from mushroom_rl.approximators.parametric import LinearApproximator, TorchApproximator
from mushroom_rl.approximators.parametric.networks import LinearNetwork
from mushroom_rl.distributions import GaussianDiagonalDistribution, GaussianCholeskyDistribution, \
    DiagonalGaussianTorchDistribution
from mushroom_rl.environments import LQR
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.rl_utils.optimizers import AdaptiveOptimizer


def learn(alg, approximator_class=LinearApproximator, distribution_class=GaussianDiagonalDistribution,
          sigma=1e-3, n_episodes=5, logger=None, **alg_params):
    np.random.seed(1)
    torch.manual_seed(1)

    # MDP
    mdp = LQR.generate(dimensions=2)

    approximator = approximator_class(input_shape=mdp.info.observation_space.shape,
                                      output_shape=mdp.info.action_space.shape)
    policy = DeterministicPolicy(approximator)
    n = policy.weights_size

    if distribution_class is DiagonalGaussianTorchDistribution:
        mu, sigma = torch.zeros(n), sigma * torch.ones(n)
    elif distribution_class is GaussianCholeskyDistribution:
        mu, sigma = np.zeros(n), sigma * np.eye(n)
    else:
        mu, sigma = np.zeros(n), sigma * np.ones(n)

    distribution = distribution_class(mu, sigma)

    agent = alg(mdp.info, distribution, policy, **alg_params)
    if logger is not None:
        agent.set_logger(logger)
    core = Core(agent, mdp)

    core.learn(n_episodes=n_episodes, n_episodes_per_fit=n_episodes)

    return agent


def learn_vectorized(alg, **alg_params):
    np.random.seed(1)
    torch.manual_seed(1)

    # MDP
    mdp = MultiprocessEnvironment(LQR, dimensions=2, n_envs=5, use_generator=True)

    approximator = LinearApproximator(input_shape=mdp.info.observation_space.shape,
                                      output_shape=mdp.info.action_space.shape)

    policy = DeterministicPolicy(mu=approximator)

    mu = np.zeros(policy.weights_size)
    sigma = 1e-3 * np.ones(policy.weights_size)
    distribution = GaussianDiagonalDistribution(mu, sigma)

    agent = alg(mdp.info, distribution, policy, **alg_params)
    core = Core(agent, mdp)

    try:
        core.learn(n_episodes=25, n_episodes_per_fit=5, quiet=True)
    finally:
        mdp.close_all()

    return agent


def test_black_box_evaluate_greedy_uses_distribution_mean():
    np.random.seed(1)
    torch.manual_seed(1)

    mdp = LQR.generate(dimensions=2)

    approximator = LinearApproximator(input_shape=mdp.info.observation_space.shape,
                                      output_shape=mdp.info.action_space.shape)
    policy = DeterministicPolicy(approximator)
    n = policy.weights_size
    mu, sigma = np.zeros(n), 1e-1 * np.ones(n)
    distribution = GaussianDiagonalDistribution(mu, sigma)

    agent = REPS(mdp.info, distribution, policy, eps=.7)
    core = Core(agent, mdp)

    dataset = core.evaluate(n_episodes=3, greedy=True, quiet=True)

    expected_theta = distribution.mean()
    for theta in dataset.theta:
        assert np.allclose(theta, expected_theta)


def test_black_box_evaluate_vectorized_greedy_uses_distribution_mean():
    np.random.seed(1)
    torch.manual_seed(1)

    mdp = MultiprocessEnvironment(LQR, dimensions=2, n_envs=5, use_generator=True)

    approximator = LinearApproximator(input_shape=mdp.info.observation_space.shape,
                                      output_shape=mdp.info.action_space.shape)
    policy = DeterministicPolicy(mu=approximator)

    mu = np.zeros(policy.weights_size)
    sigma = 1e-1 * np.ones(policy.weights_size)
    distribution = GaussianDiagonalDistribution(mu, sigma)

    agent = REPS(mdp.info, distribution, policy, eps=.7)
    core = Core(agent, mdp)

    try:
        dataset = core.evaluate(n_episodes=5, greedy=True, quiet=True)
    finally:
        mdp.close_all()

    expected_theta = distribution.mean()
    for theta in dataset.theta:
        assert np.allclose(theta, expected_theta)


def test_RWR(tmpdir):
    agent = learn(RWR, beta=1.)
    distribution = agent.distribution
    w = distribution.get_parameters()
    w_test = np.load('tests/algorithms/data/test_RWR.npy')

    assert np.allclose(w, w_test)

    tu.assert_save_load(agent, tmpdir / 'agent.msh')


def test_logger():
    logger = FakeLogger()
    agent = learn(RWR, beta=1., logger=logger)

    assert np.allclose(agent.distribution.get_parameters(), np.load('tests/algorithms/data/test_RWR.npy'))
    assert logger.keys() == {'J', 'distribution/entropy'} and len(logger.calls) == 2

    with pytest.raises(NotImplementedError):
        BlackBoxOptimization._update(None, None, None, None)


def test_REPS(tmpdir):
    agent = learn(REPS, eps=.7)
    distribution = agent.distribution
    w = distribution.get_parameters()
    w_test = np.load('tests/algorithms/data/test_REPS.npy')

    assert np.allclose(w, w_test)

    tu.assert_save_load(agent, tmpdir / 'agent.msh')


def test_ConstrainedREPS(tmpdir):
    agent = learn(ConstrainedREPS, eps=.7, kappa=0.1)
    distribution = agent.distribution
    w = distribution.get_parameters()
    w_test = np.load('tests/algorithms/data/test_ConstrainedREPS.npy')

    assert np.allclose(w, w_test)

    tu.assert_save_load(agent, tmpdir / 'agent.msh')


def test_RWR_vectorized():
    distribution = learn_vectorized(RWR, beta=1.).distribution
    w = distribution.get_parameters()
    w_test = np.load('tests/algorithms/data/test_RWR_vectorized.npy')

    assert np.allclose(w, w_test)


def test_REPS_vectorized():
    distribution = learn_vectorized(REPS, eps=.7).distribution
    w = distribution.get_parameters()
    w_test = np.load('tests/algorithms/data/test_REPS_vectorized.npy')

    assert np.allclose(w, w_test)


def test_PGPE(tmpdir):
    agent = learn(PGPE, optimizer=AdaptiveOptimizer(1.5))
    distribution = agent.distribution
    w = distribution.get_parameters()
    w_test = np.load('tests/algorithms/data/test_PGPE.npy')

    assert np.allclose(w, w_test)

    tu.assert_save_load(agent, tmpdir / 'agent.msh')


def test_PGPE_vectorized():
    distribution = learn_vectorized(PGPE, optimizer=AdaptiveOptimizer(1.5)).distribution
    w = distribution.get_parameters()
    w_test = np.load('tests/algorithms/data/test_PGPE_vectorized.npy')

    assert np.allclose(w, w_test)


def test_MORE(tmpdir):
    agent = learn(MORE, distribution_class=GaussianCholeskyDistribution, sigma=1e-1, n_episodes=100, eps=0.5,
                  kappa=0.99)
    w = agent.distribution.get_parameters()
    w_test = np.load('tests/algorithms/data/test_MORE.npy')

    assert np.allclose(w, w_test)

    tu.assert_save_load(agent, tmpdir / 'agent.msh')


@pytest.mark.parametrize('device', devices())
def test_ePPO(device, tmpdir):
    optimizer = {'class': optim.Adam, 'params': {'lr': 1e-2, 'weight_decay': 0.0}}
    agent = learn(ePPO, approximator_class=partial(TorchApproximator, network=LinearNetwork),
                  distribution_class=DiagonalGaussianTorchDistribution, sigma=1e-1,
                  optimizer=optimizer, n_epochs_policy=5, batch_size=5, eps_ppo=5e-2)
    tu.assert_recorded(agent.distribution.get_parameters().detach(), 'test_ePPO', device)

    tu.assert_save_load(agent, tmpdir / 'agent.msh')
