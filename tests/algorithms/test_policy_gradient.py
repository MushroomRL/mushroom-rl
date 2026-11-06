import numpy as np
import pytest
import torch

from helper.dummies import FakeLogger
from helper.utils import TestUtils as tu

from mushroom_rl.algorithms.policy_search import REINFORCE, GPOMDP, eNAC
from mushroom_rl.algorithms.policy_search.policy_gradient import PolicyGradient
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.core import Core
from mushroom_rl.environments.lqr import LQR
from mushroom_rl.policy.gaussian_policy import StateStdGaussianPolicy
from mushroom_rl.rl_utils.optimizers import AdaptiveOptimizer


def learn(alg, alg_params, logger=None):
    mdp = LQR.generate(dimensions=1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    approximator = LinearApproximator(input_shape=mdp.info.observation_space.shape,
                                      output_shape=mdp.info.action_space.shape)

    sigma = LinearApproximator(input_shape=mdp.info.observation_space.shape,
                               output_shape=mdp.info.action_space.shape)

    sigma_weights = 2 * np.ones(sigma.weights_size)
    sigma.set_weights(sigma_weights)

    policy = StateStdGaussianPolicy(approximator, sigma)

    agent = alg(mdp.info, policy, **alg_params)
    if logger is not None:
        agent.set_logger(logger)

    core = Core(agent, mdp)

    core.learn(n_episodes=10, n_episodes_per_fit=5)

    return agent


def test_REINFORCE(tmpdir):
    logger = FakeLogger()
    params = dict(optimizer=AdaptiveOptimizer(eps=.01))
    agent = learn(REINFORCE, params, logger=logger)
    policy = agent.policy
    w = np.array([-0.0084793,  2.00536528])

    assert np.allclose(w, policy.get_weights())
    assert logger.keys() == {'J'} and len(logger.calls) == 2

    tu.assert_save_load(agent, tmpdir / 'agent.msh')


def test_policy_gradient():
    for call in (lambda: PolicyGradient._init_update(None), lambda: PolicyGradient._step_update(None, 0, 0, 0),
                 lambda: PolicyGradient._episode_end_update(None), lambda: PolicyGradient._compute_gradient(None, 0)):
        with pytest.raises(NotImplementedError):
            call()


def test_GPOMDP(tmpdir):
    params = dict(optimizer=AdaptiveOptimizer(eps=.01))
    agent = learn(GPOMDP, params)
    policy = agent.policy
    w = np.array([-0.11457566, 1.99784316])

    assert np.allclose(w, policy.get_weights())

    tu.assert_save_load(agent, tmpdir / 'agent.msh')


def test_eNAC(tmpdir):
    params = dict(optimizer=AdaptiveOptimizer(eps=.01))
    agent = learn(eNAC, params)
    policy = agent.policy
    w = np.array([-0.16169364,  2.00594995])

    assert np.allclose(w, policy.get_weights())

    tu.assert_save_load(agent, tmpdir / 'agent.msh')
