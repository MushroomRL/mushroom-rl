import numpy as np
import torch
from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.core import Agent
from mushroom_rl.algorithms.policy_search import *
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.core import Core
from mushroom_rl.environments.lqr import LQR
from mushroom_rl.policy.gaussian_policy import StateStdGaussianPolicy
from mushroom_rl.rl_utils.optimizers import AdaptiveOptimizer


def learn(alg, alg_params):
    mdp = LQR.generate(dimensions=1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

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

    agent = alg(mdp.info, policy, **alg_params)

    core = Core(agent, mdp)

    core.learn(n_episodes=10, n_episodes_per_fit=5)

    return agent


def test_REINFORCE():
    params = dict(optimizer=AdaptiveOptimizer(eps=.01))
    policy = learn(REINFORCE, params).policy
    w = np.array([-0.0084793 ,  2.00536528])

    assert np.allclose(w, policy.get_weights())


def test_REINFORCE_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(optimizer=AdaptiveOptimizer(eps=.01))

    agent_save = learn(REINFORCE, params)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_GPOMDP():
    params = dict(optimizer=AdaptiveOptimizer(eps=.01))
    policy = learn(GPOMDP, params).policy
    w = np.array([-0.11457566, 1.99784316])

    assert np.allclose(w, policy.get_weights())


def test_GPOMDP_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(optimizer=AdaptiveOptimizer(eps=.01))

    agent_save = learn(GPOMDP, params)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_eNAC():
    params = dict(optimizer=AdaptiveOptimizer(eps=.01))
    policy = learn(eNAC, params).policy
    w = np.array([-0.16169364,  2.00594995])

    assert np.allclose(w, policy.get_weights())


def test_eNAC_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(optimizer=AdaptiveOptimizer(eps=.01))

    agent_save = learn(eNAC, params)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)
