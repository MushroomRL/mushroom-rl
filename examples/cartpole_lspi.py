import numpy as np

from mushroom_rl.algorithms.value import LSPI
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import *
from mushroom_rl.features import Features
from mushroom_rl.features.basis import PolynomialBasis, GaussianRBF
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter


"""
This script aims to replicate the experiments on the Inverted Pendulum MDP as
presented in:
"Least-Squares Policy Iteration". Lagoudakis M. G. and Parr R.. 2003.

"""


def experiment():
    np.random.seed()

    # MDP
    mdp = CartPole()

    # Policy
    epsilon = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    basis = [PolynomialBasis()]

    s1 = np.array([-np.pi, 0, np.pi]) * .25
    s2 = np.array([-1, 0, 1])
    for i in s1:
        for j in s2:
            basis.append(GaussianRBF(np.array([i, j]), np.array([1.])))
    features = Features(basis_list=basis)

    fit_params = dict()
    approximator_params = dict(input_shape=(features.size,),
                               output_shape=(mdp.info.action_space.n,),
                               n_actions=mdp.info.action_space.n,
                               phi=features)
    agent = LSPI(mdp.info, pi, approximator_params=approximator_params, fit_params=fit_params)

    # Algorithm
    core = Core(agent, mdp)
    core.evaluate(n_episodes=3, render=True)

    # Train
    core.learn(n_episodes=500, n_episodes_per_fit=500)

    # Test
    test_epsilon = Parameter(0.)
    agent.policy.set_epsilon(test_epsilon)

    dataset = core.evaluate(n_episodes=1, quiet=True)

    core.evaluate(n_steps=100, render=True)

    return np.mean(dataset.episodes_length)


if __name__ == '__main__':
    n_experiment = 1

    logger = Logger(LSPI.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + LSPI.__name__)

    steps = experiment()
    logger.info('Final episode length: %d' % steps)
