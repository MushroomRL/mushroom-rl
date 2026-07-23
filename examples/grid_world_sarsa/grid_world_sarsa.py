import argparse
from pathlib import Path

import numpy as np

from mushroom_rl.algorithms.value import SARSALambda
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import GridWorld, Taxi
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter
from mushroom_rl.solvers.dynamic_programming import value_iteration


"""
Simple script to solve a grid world or the taxi problem with SARSA(lambda).

Both environments are finite MDPs, so the return of the optimal policy can be computed exactly with value iteration
and used as a reference for the learned one. The grid world needs no eligibility trace, and is solved by the plain
SARSA recovered with lambda set to zero. The taxi problem does: the goal reward has to travel back past the pick ups,
and a one step update leaves the agent collecting a single passenger. A long trace carries the reward along the whole
trajectory, and the optimal route is found on a good share of the runs.

"""


def experiment(mdp, n_steps, lambda_coeff):
    logger = Logger(type(mdp).__name__, results_dir=None)
    logger.strong_line()
    logger.info('Environment: ' + type(mdp).__name__)
    logger.info('Experiment Algorithm: ' + SARSALambda.__name__)

    # Policy
    epsilon = Parameter(value=.1)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = Parameter(value=.1)
    agent = SARSALambda(mdp.info, pi, learning_rate, lambda_coeff)

    # Core
    core = Core(agent, mdp)

    # Optimal return, computed by dynamic programming on the transition and reward matrices
    value_optimal = value_iteration(mdp.p, mdp.r, mdp.info.gamma, 1e-8)
    logger.info(f'J optimal: {mdp.iota.dot(value_optimal)}')

    # Visualize initial policy
    core.evaluate(n_steps=50, render=True)

    # Initial policy evaluation
    dataset = core.evaluate(n_episodes=100, greedy=True)
    logger.info(f'J start: {np.mean(dataset.discounted_return)}')

    # Train
    core.learn(n_steps=n_steps, n_steps_per_fit=1)

    # Final policy evaluation
    dataset = core.evaluate(n_episodes=100, greedy=True)
    logger.info(f'J final: {np.mean(dataset.discounted_return)}')

    # Visualize final policy
    core.evaluate(n_steps=50, render=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['grid_world', 'taxi'], default='grid_world',
                        help='the environment to solve')
    args = parser.parse_args()

    np.random.seed()

    maps = Path(__file__).resolve().parent

    if args.env == 'grid_world':
        mdp, n_steps, lambda_coeff = GridWorld.from_file(maps / 'grid.txt', prob=.9), 20000, 0.
    else:
        mdp, n_steps, lambda_coeff = Taxi.from_file(maps / 'taxi.txt', goal_rewards=(0, 1, 5)), 100000, .95

    experiment(mdp, n_steps, lambda_coeff)
