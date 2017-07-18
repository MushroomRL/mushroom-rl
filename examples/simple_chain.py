import numpy as np

from PyPi.algorithms.td import QLearning
from PyPi.approximators import Regressor, Tabular
from PyPi.core.core import Core
from PyPi.environments import *
from PyPi.policy import EpsGreedy
from PyPi.utils.parameters import Parameter


def experiment():
    np.random.seed()

    # MDP
    mdp = generate_simple_chain(state_n=5, goal_states=[2], prob=.8, rew=1,
                                gamma=0.9)

    # Policy
    epsilon = Parameter(value=.15)
    pi = EpsGreedy(epsilon=epsilon, observation_space=mdp.observation_space,
                   action_space=mdp.action_space)

    # Approximator
    shape = mdp.observation_space.shape + mdp.action_space.shape
    approximator_params = dict(shape=shape)
    approximator = Regressor(Tabular, **approximator_params)

    # Agent
    learning_rate = Parameter(value=.2)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = QLearning(approximator, pi, **agent_params)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_iterations=10000, how_many=1, n_fit_steps=1,
               iterate_over='samples')

if __name__ == '__main__':
    out = experiment()
