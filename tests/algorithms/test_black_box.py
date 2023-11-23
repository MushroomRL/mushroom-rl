import numpy as np
import torch
from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.core import Agent
from mushroom_rl.algorithms.policy_search import PGPE, REPS, ConstrainedREPS, RWR
from mushroom_rl.approximators import Regressor
from mushroom_rl.core import Core
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.distributions import GaussianDiagonalDistribution
from mushroom_rl.environments import LQR
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.rl_utils.optimizers import AdaptiveOptimizer


def learn(alg, **alg_params):
    np.random.seed(1)
    torch.manual_seed(1)

    # MDP
    mdp = LQR.generate(dimensions=2)

    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape)

    policy = DeterministicPolicy(mu=approximator)

    mu = np.zeros(policy.weights_size)
    sigma = 1e-3 * np.ones(policy.weights_size)
    distribution = GaussianDiagonalDistribution(mu, sigma)

    agent = alg(mdp.info, distribution, policy, **alg_params)
    core = Core(agent, mdp)

    core.learn(n_episodes=5, n_episodes_per_fit=5)

    return agent


def test_RWR():
    distribution = learn(RWR, beta=1.).distribution
    w = distribution.get_parameters()
    w_test = np.array([0.00086195, -0.00229678, 0.00173919, -0.0007568,
                       0.00073533, 0.00101203, 0.00119701, 0.00094453])

    assert np.allclose(w, w_test)


def test_RWR_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    agent_save = learn(RWR, beta=1.)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_REPS():
    distribution = learn(REPS, eps=.7).distribution
    w = distribution.get_parameters()
    w_test = np.array([0.00050246, -0.00175432, 0.00128979, -0.00050779,
                       0.00071795, 0.00108254, 0.00098966, 0.00086633])

    assert np.allclose(w, w_test)


def test_REPS_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    agent_save = learn(REPS, eps=.7)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_ConstrainedREPS():
    distribution = learn(ConstrainedREPS, eps=.7, kappa=0.1).distribution
    w = distribution.get_parameters()
    w_test = np.array([0.00026583, -0.00092814,  0.00068238, -0.00026865,
                       0.00081971, 0.00124566,  0.00107135,  0.00085804])

    assert np.allclose(w, w_test)


def test_ConstrainedREPS_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    agent_save = learn(ConstrainedREPS, eps=.7, kappa=0.1)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_PGPE():
    distribution = learn(PGPE, optimizer=AdaptiveOptimizer(1.5)).distribution
    w = distribution.get_parameters()
    w_test = np.array([0.02489092, 0.31062211, 0.2051433, 0.05959651,
                       -0.78302236, 0.77381954, 0.23676176, -0.29855654])

    assert np.allclose(w, w_test)


def test_PGPE_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    agent_save = learn(PGPE, optimizer=AdaptiveOptimizer(1.5))

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)
