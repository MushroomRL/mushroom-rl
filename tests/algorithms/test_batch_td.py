import numpy as np

from mushroom_rl.algorithms.value import LSPI
from mushroom_rl.core import Core
from mushroom_rl.environments import *
from mushroom_rl.features import Features
from mushroom_rl.features.basis import PolynomialBasis
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter


def test_lspi():
    mdp = CartPole()
    np.random.seed(1)

    # Policy
    epsilon = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    basis = [PolynomialBasis()]
    features = Features(basis_list=basis)

    approximator_params = dict(input_shape=(features.size,),
                               output_shape=(mdp.info.action_space.n,),
                               n_actions=mdp.info.action_space.n)
    agent = LSPI(mdp.info, pi, fit_params=dict(),
                 approximator_params=approximator_params, features=features)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_episodes=100, n_episodes_per_fit=100)

    w = agent.approximator.get_weights()
    w_test = np.array([-2.23880597, -2.27427603, -2.25])

    assert np.allclose(w, w_test)
