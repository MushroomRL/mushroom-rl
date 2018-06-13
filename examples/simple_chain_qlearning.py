import numpy as np

from mushroom.algorithms.value import QLearning
from mushroom.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.parameters import Parameter


"""
Simple script to solve a simple chain with Q-Learning.

"""


def experiment():
    np.random.seed()

    # MDP
    mdp = generate_simple_chain(state_n=5, goal_states=[2], prob=.8, rew=1,
                                gamma=.9)

    # Policy
    epsilon = Parameter(value=.15)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = Parameter(value=.2)
    algorithm_params = dict(learning_rate=learning_rate)
    agent = QLearning(pi, mdp.info, **algorithm_params)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=10000, n_steps_per_fit=1)


if __name__ == '__main__':
    experiment()
