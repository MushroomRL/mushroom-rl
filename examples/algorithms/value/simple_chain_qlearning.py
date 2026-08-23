"""
Simple script to solve a simple chain with Q-Learning.

"""
import numpy as np

from mushroom_rl.algorithms.value import QLearning
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import SimpleChain
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter


def experiment(n_steps, seed=None):
    np.random.seed(seed)

    # MDP
    mdp = SimpleChain(n_states=5, goal_states=[2], prob=.8, goal_reward=1, gamma=.9)

    logger = Logger(QLearning.name(), results_dir=None)
    logger.log_experiment_info(QLearning, mdp, n_steps=n_steps)

    # Policy
    epsilon = Parameter(value=.15)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = Parameter(value=.2)
    agent = QLearning(mdp.info, pi, learning_rate=learning_rate)

    # Core
    core = Core(agent, mdp, logger=logger)

    # Visualize initial policy
    core.evaluate(n_steps=100, render=True)

    # Initial policy Evaluation
    dataset = core.evaluate(n_steps=1000)
    logger.info(f'J start: {dataset.discounted_return.mean()}')

    # Train
    core.learn(n_steps=n_steps, n_steps_per_fit=1)

    # Final Policy Evaluation
    dataset = core.evaluate(n_steps=1000)
    logger.info(f'J final: {dataset.discounted_return.mean()}')

    # Visualize final policy
    core.evaluate(n_steps=100, render=True)


if __name__ == '__main__':
    experiment(n_steps=10000)
