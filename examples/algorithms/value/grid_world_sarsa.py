"""
Simple script to solve a grid world or the taxi problem with SARSA(lambda).

Both environments are finite MDPs, so the return of the optimal policy can be computed exactly with value iteration
and used as a reference for the learned one.
The grid world needs no eligibility trace, and is solved by the plain SARSA (lambda set to zero).
The taxi problem requires eligibility trace: the goal reward has to travel back past the pick-ups, and a one-step update
leaves the agent collecting a single passenger. A long trace carries the reward along the whole trajectory, and the
optimal route is found on a good share of the runs.

"""
import argparse

import numpy as np

from mushroom_rl.algorithms.value import SARSALambda
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import GridWorld, Taxi
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter
from mushroom_rl.solvers.dynamic_programming import value_iteration
from mushroom_rl.utils import get_data_dir


def build_mdp(env):
    """
    Build the requested environment from its map, together with the length of the run and the trace decay it
    needs to be solved.

    """
    maps = get_data_dir(__file__) / 'grid_world'

    if env == 'grid_world':
        return GridWorld.from_file(maps / 'grid.txt', prob=.9), 20000, 0.
    else:
        return Taxi.from_file(maps / 'taxi.txt', goal_rewards=(0, 1, 5)), 100000, .95


def experiment(env, seed=None):
    np.random.seed(seed)

    # MDP
    mdp, n_steps, lambda_coeff = build_mdp(env)

    logger = Logger(mdp.name(), results_dir=None)
    logger.log_experiment_info(SARSALambda, mdp, n_steps=n_steps, lambda_coeff=lambda_coeff)

    # Policy
    epsilon = Parameter(value=.1)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = Parameter(value=.1)
    agent = SARSALambda(mdp.info, pi, learning_rate, lambda_coeff)

    # Core
    core = Core(agent, mdp, logger=logger)

    # Optimal return, computed by dynamic programming on the transition and reward matrices
    value_optimal = value_iteration(mdp.p, mdp.r, mdp.info.gamma, 1e-8)
    logger.info(f'J optimal: {mdp.iota.dot(value_optimal)}')

    # Visualize initial policy
    core.evaluate(n_steps=50, render=True)

    # Initial policy evaluation
    dataset = core.evaluate(n_episodes=100, greedy=True)
    logger.info(f'J start: {dataset.discounted_return.mean()}')

    # Train
    core.learn(n_steps=n_steps, n_steps_per_fit=1)

    # Final policy evaluation
    dataset = core.evaluate(n_episodes=100, greedy=True)
    logger.info(f'J final: {dataset.discounted_return.mean()}')

    # Visualize final policy
    core.evaluate(n_steps=50, render=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', choices=['grid_world', 'taxi'], default='grid_world', help='the environment to solve')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    experiment(args.env)
