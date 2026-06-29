import numpy as np
import torch
from torch import optim
from functools import partial
from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.core import Agent
from mushroom_rl.algorithms.policy_search import PGPE, REPS, ConstrainedREPS, RWR, MORE, ePPO
from mushroom_rl.core import Core, MultiprocessEnvironment
from mushroom_rl.approximators.parametric import LinearApproximator, TorchApproximator
from mushroom_rl.approximators.parametric.networks import LinearNetwork
from mushroom_rl.distributions import GaussianDiagonalDistribution, GaussianCholeskyDistribution,\
    DiagonalGaussianTorchDistribution
from mushroom_rl.environments import LQR
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.rl_utils.optimizers import AdaptiveOptimizer


def learn(alg, approximator_class=LinearApproximator, distribution_class=GaussianDiagonalDistribution,
          sigma=1e-3, n_episodes=5, **alg_params):
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


def test_RWR_vectorized():
    distribution = learn_vectorized(RWR, beta=1.).distribution
    w = distribution.get_parameters()
    w_test = np.array([-4.69344455e-04, -2.89837445e-03, 3.14330005e-03, -1.16009810e-03,
                       1.08396911e-04, 2.82932526e-05, 2.13583517e-04, 8.57529937e-05])

    assert np.allclose(w, w_test)


def test_REPS_vectorized():
    distribution = learn_vectorized(REPS, eps=.7).distribution
    w = distribution.get_parameters()
    w_test = np.array([3.38656856e-04, -2.65343883e-03, 1.22095958e-03, -1.29026211e-03,
                       9.23115364e-05, 3.78374156e-05, 4.76125495e-04, 1.95123615e-04])

    assert np.allclose(w, w_test)


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


def test_PGPE_vectorized():
    distribution = learn_vectorized(PGPE, optimizer=AdaptiveOptimizer(1.5)).distribution
    w = distribution.get_parameters()
    w_test = np.array([-0.98513892, -0.07076845, 0.53698872, -0.75166881,
                       0.2858429, 0.4656616, 0.98251223, -0.22922507])

    assert np.allclose(w, w_test)


def test_MORE():
    distribution = learn(MORE, distribution_class=GaussianCholeskyDistribution, sigma=1e-1,
                         n_episodes=100, eps=0.5, kappa=0.99).distribution
    w = distribution.get_parameters()
    w_test = np.array([-0.04363140, 0.01242653, 0.01254028, -0.06094352, 0.28314392,
                       -0.02048622, 0.29749173, 0.17302031, -0.00197523, 0.17936346,
                       -0.04546476, -0.04406964, 0.10505358, 0.30932095])

    assert np.allclose(w, w_test)


def test_MORE_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    agent_save = learn(MORE, distribution_class=GaussianCholeskyDistribution, sigma=1e-1,
                       n_episodes=100, eps=0.5, kappa=0.99)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_ePPO():
    optimizer = {'class': optim.Adam, 'params': {'lr': 1e-2, 'weight_decay': 0.0}}
    distribution = learn(ePPO, approximator_class=partial(TorchApproximator, network=LinearNetwork),
                         distribution_class=DiagonalGaussianTorchDistribution, sigma=1e-1,
                         optimizer=optimizer, n_epochs_policy=5, batch_size=5, eps_ppo=5e-2).distribution
    w = distribution.get_parameters().detach().numpy()
    w_test = np.array([-0.02981966, 0.01895342, -0.02684142, 0.03063847,
                       -2.33322644, -2.33125615, -2.33870649, -2.33909726], dtype=np.float32)

    assert np.allclose(w, w_test)


def test_ePPO_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    optimizer = {'class': optim.Adam, 'params': {'lr': 1e-2, 'weight_decay': 0.0}}
    agent_save = learn(ePPO, approximator_class=partial(TorchApproximator, network=LinearNetwork),
                       distribution_class=DiagonalGaussianTorchDistribution, sigma=1e-1,
                       optimizer=optimizer, n_epochs_policy=5, batch_size=5, eps_ppo=5e-2)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)
