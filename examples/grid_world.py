import numpy as np
from joblib import Parallel, delayed

from PyPi.algorithms.td import QLearning, DoubleQLearning
from PyPi.approximators import Ensemble, Regressor, Tabular
from PyPi.core.core import Core
from PyPi.environments import *
from PyPi.policy import EpsGreedy
from PyPi.utils import logger
from PyPi.utils.dataset import parse_dataset
from PyPi.utils.parameters import Parameter


def experiment():
    np.random.seed()

    # MDP
    mdp = GridWorldVanHasselt()

    # Policy
    epsilon = Parameter(value=1, decay=True, decay_exp=.5,
                        shape=mdp.observation_space.shape)
    pi = EpsGreedy(epsilon=epsilon, observation_space=mdp.observation_space,
                   action_space=mdp.action_space)

    # Approximator
    approximator_params = dict(shape=(mdp.observation_space.shape +
                                      mdp.action_space.shape))
    approximator = Ensemble(Tabular, 2, **approximator_params)

    # Agent
    learning_rate = Parameter(value=1, decay=True, decay_exp=1,
                              shape=(mdp.observation_space.shape +
                                     mdp.action_space.shape))
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = DoubleQLearning(approximator, pi, **agent_params)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_iterations=1, how_many=500000, n_fit_steps=1,
               iterate_over='samples')

    _, _, reward, _, _, _ = parse_dataset(core.get_dataset(),
                                          mdp.observation_space.dim,
                                          mdp.action_space.dim)
    return reward

if __name__ == '__main__':
    n_experiment = 2

    logger.Logger(3)

    reward = Parallel(n_jobs=-1)(delayed(experiment)() for _ in range(n_experiment))

    from matplotlib import pyplot as plt
    plt.plot(np.mean(reward, axis=0))
    plt.show()
