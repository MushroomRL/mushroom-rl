import numpy as np

from mushroom_rl.algorithms.value import QLearning
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter


"""
Simple script to solve a simple chain with Q-Learning.

"""


def experiment():
    np.random.seed()

    logger = Logger(QLearning.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + QLearning.__name__)

    # MDP
    mdp = generate_simple_chain(state_n=5, goal_states=[2], prob=.8, rew=1,
                                gamma=.9)

    # Policy
    epsilon = Parameter(value=.15)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = Parameter(value=.2)
    algorithm_params = dict(learning_rate=learning_rate)
    agent = QLearning(mdp.info, pi, **algorithm_params)

    # Core
    core = Core(agent, mdp)

    # Initial policy Evaluation
    dataset = core.evaluate(n_steps=1000)
    J = np.mean(dataset.discounted_return)
    logger.info(f'J start: {J}')

    # Train
    core.learn(n_steps=10000, n_steps_per_fit=1)

    # Final Policy Evaluation
    dataset = core.evaluate(n_steps=1000)
    J = np.mean(dataset.discounted_return)
    logger.info(f'J final: {J}')


if __name__ == '__main__':
    experiment()
