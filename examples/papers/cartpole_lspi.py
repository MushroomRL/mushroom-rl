"""
This script aims to replicate the experiments on the Inverted Pendulum MDP as presented in:
"Least-Squares Policy Iteration". Lagoudakis M. G. and Parr R. 2003.

"""
import numpy as np

from mushroom_rl.algorithms.value import LSPI
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import CartPole
from mushroom_rl.features import Features
from mushroom_rl.features.basis import PolynomialBasis, GaussianRBF
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter


def experiment(seed=0):
    np.random.seed(seed)

    # MDP
    mdp = CartPole()

    logger = Logger(LSPI.name(), results_dir=None)
    logger.log_experiment_info(LSPI, mdp)

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
    features = Features(basis)

    approximator_params = dict(input_shape=mdp.info.observation_space.shape,
                               output_shape=(mdp.info.action_space.n,),
                               n_actions=mdp.info.action_space.n,
                               phi=features)
    agent = LSPI(mdp.info, pi, approximator_params=approximator_params, fit_params=dict())

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    # Visualize the initial policy
    core.evaluate(n_episodes=3, render=True)

    # Train
    core.learn(n_episodes=500, n_episodes_per_fit=500)

    # Test
    dataset = core.evaluate(n_episodes=1, quiet=True, greedy=True)
    logger.info(f'Final episode length: {dataset.episodes_length.mean()}')

    # Visualize the final policy
    core.evaluate(n_steps=100, render=True, greedy=True)


if __name__ == '__main__':
    experiment()
