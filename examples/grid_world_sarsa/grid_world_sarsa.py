import argparse

import numpy as np

from mushroom_rl.algorithms.value import SARSA
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import GridWorld, Taxi
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter
from mushroom_rl.solvers.dynamic_programming import value_iteration


"""
Simple script to solve a grid world or the taxi problem with SARSA.

Both environments are finite MDPs, so the return of the optimal policy can be computed exactly with value iteration
and used as a reference for the learned one. On the taxi problem SARSA settles for a much lower return than the
optimal one: picking up every passenger before reaching the goal is hard to discover with a fixed exploration rate.

"""


def experiment(mdp, n_steps):
    logger = Logger(type(mdp).__name__, results_dir=None)
    logger.strong_line()
    logger.info('Environment: ' + type(mdp).__name__)
    logger.info('Experiment Algorithm: ' + SARSA.__name__)

    # Policy
    epsilon = Parameter(value=.1)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = Parameter(value=.1)
    agent = SARSA(mdp.info, pi, learning_rate)

    # Core
    core = Core(agent, mdp)

    # Optimal return, computed by dynamic programming on the transition and reward matrices
    value_optimal = value_iteration(mdp.p, mdp.r, mdp.info.gamma, 1e-8)
    logger.info(f'J optimal: {mdp.mu.dot(value_optimal)}')

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

    if args.env == 'grid_world':
        mdp, n_steps = GridWorld.from_file('grid.txt', prob=.9), 20000
    else:
        mdp, n_steps = Taxi.from_file('taxi.txt', goal_rewards=(0, 1, 5)), 100000

    experiment(mdp, n_steps)
