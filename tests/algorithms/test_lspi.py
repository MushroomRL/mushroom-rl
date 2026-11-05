import numpy as np

from helper.dummies import FakeLogger
from helper.utils import TestUtils as tu

from mushroom_rl.algorithms.value import LSPI
from mushroom_rl.core import Core
from mushroom_rl.environments import CartPole
from mushroom_rl.features import Features
from mushroom_rl.features.basis import PolynomialBasis
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter


def learn_lspi(logger=None):
    np.random.seed(1)

    # MDP
    mdp = CartPole()

    # Policy
    epsilon = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    basis = [PolynomialBasis()]
    features = Features(basis)

    fit_params = dict()
    approximator_params = dict(input_shape=mdp.info.observation_space.shape,
                               output_shape=(mdp.info.action_space.n,),
                               n_actions=mdp.info.action_space.n,
                               phi=features)
    agent = LSPI(mdp.info, pi, approximator_params=approximator_params, fit_params=fit_params)
    if logger is not None:
        agent.set_logger(logger)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_episodes=10, n_episodes_per_fit=10)

    return agent


def test_lspi(tmpdir):
    agent = learn_lspi()
    w = agent.approximator.get_weights()
    w_test = np.array([-1.67115903, -1.43755615, -1.67115903])

    assert np.allclose(w, w_test)

    tu.assert_save_load(agent, tmpdir / 'agent.msh')


def test_logger():
    logger = FakeLogger()
    agent = learn_lspi(logger)

    assert np.allclose(agent.approximator.get_weights(), np.array([-1.67115903, -1.43755615, -1.67115903]))
    assert logger.keys() == {'critic/weights_delta_norm', 'policy/epsilon'}
