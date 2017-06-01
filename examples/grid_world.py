import numpy as np
from joblib import Parallel, delayed

from PyPi.algorithms.td import QLearning, DoubleQLearning
from PyPi.approximators import Ensemble, Regressor, Tabular
from PyPi.core.core import Core
from PyPi.environments import *
from PyPi.policy import EpsGreedy
from PyPi.utils import logger
from PyPi.utils.parameters import Parameter


def experiment():
    np.random.seed()

    # MDP
    mdp = GridWorld(height=3, width=3, goal=(2, 2))

    # Policy
    epsilon = Parameter(value=1)
    discrete_actions = mdp.action_space.values
    pi = EpsGreedy(epsilon=epsilon, discrete_actions=discrete_actions)

    # Approximator
    approximator_params = dict(shape=(3, 3, mdp.action_space.n))
    approximator = Ensemble(Tabular, 2, **approximator_params)

    # Agent
    algorithm_params = dict(learning_rate=Parameter(0.8))
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = DoubleQLearning(approximator, pi, **agent_params)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_iterations=300, how_many=1, n_fit_steps=1,
               iterate_over='samples')

    # Test
    test_epsilon = Parameter(0)
    agent.policy.set_epsilon(test_epsilon)

    initial_states = np.array([[0, 0]])

    return np.mean(core.evaluate(initial_states, render=False))


if __name__ == '__main__':
    n_experiment = 1

    logger.Logger(0)

    Js = Parallel(n_jobs=-1)(delayed(experiment)() for _ in range(n_experiment))
    print(Js)
