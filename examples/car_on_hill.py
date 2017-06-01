import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor

from PyPi.algorithms.batch_td import FQI
from PyPi.approximators import ActionRegressor
from PyPi.core.core import Core
from PyPi.environments import *
from PyPi.policy import EpsGreedy
from PyPi.utils import logger
from PyPi.utils.parameters import Parameter


def experiment():
    np.random.seed()

    # MDP
    mdp = CarOnHill()

    # Policy
    epsilon = Parameter(value=1)
    discrete_actions = mdp.action_space.values
    pi = EpsGreedy(epsilon=epsilon, discrete_actions=discrete_actions)

    # Approximator
    approximator_params = dict()
    approximator = ActionRegressor(ExtraTreesRegressor,
                                   discrete_actions=discrete_actions,
                                   **approximator_params)

    # Agent
    algorithm_params = dict()
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = FQI(approximator, pi, **agent_params)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_iterations=1, how_many=1000, n_fit_steps=20,
               iterate_over='episodes')

    # Test
    test_epsilon = Parameter(0)
    agent.policy.set_epsilon(test_epsilon)

    initial_states = np.zeros((289, 2))
    cont = 0
    for i in range(-8, 9):
        for j in range(-8, 9):
            initial_states[cont, :] = [0.125 * i, 0.375 * j]
            cont += 1

    return np.mean(core.evaluate(initial_states, render=False))


if __name__ == '__main__':
    n_experiment = 1

    logger.Logger(1)

    Js = Parallel(n_jobs=-1)(delayed(experiment)() for _ in range(n_experiment))
    print(Js)
