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


def experiment(algorithm_class, decay_exp):
    np.random.seed()

    # MDP
    mdp = GridWorldVanHasselt()

    # Policy
    epsilon = Parameter(value=1, decay=True, decay_exp=.5,
                        shape=mdp.observation_space.shape)
    pi = EpsGreedy(epsilon=epsilon, observation_space=mdp.observation_space,
                   action_space=mdp.action_space)

    # Approximator
    shape = mdp.observation_space.shape + [mdp.action_space.shape]
    approximator_params = dict(shape=shape)
    if algorithm_class is QLearning:
        approximator = Regressor(Tabular, **approximator_params)
    elif algorithm_class is DoubleQLearning:
        approximator = Ensemble(Tabular, 2, **approximator_params)

    # Agent
    learning_rate = Parameter(value=1, decay=True, decay_exp=decay_exp,
                              shape=shape)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = algorithm_class(approximator, pi, **agent_params)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_iterations=10000, how_many=1, n_fit_steps=1,
               iterate_over='samples')

    _, _, reward, _, _, _ = parse_dataset(core.get_dataset())

    return reward

if __name__ == '__main__':
    n_experiment = 10000

    logger.Logger(1)

    rewardQ1 = Parallel(n_jobs=-1)(
        delayed(experiment)(QLearning, 1) for _ in xrange(n_experiment))
    rewardQ08 = Parallel(n_jobs=-1)(
        delayed(experiment)(QLearning, .8) for _ in xrange(n_experiment))
    rewardDQ1 = Parallel(n_jobs=-1)(
        delayed(experiment)(DoubleQLearning, 1) for _ in xrange(n_experiment))
    rewardDQ08 = Parallel(n_jobs=-1)(
        delayed(experiment)(DoubleQLearning, .8) for _ in xrange(n_experiment))

    np.save('rQ1.npy', np.mean(rewardQ1, axis=0))
    np.save('rQ08.npy', np.mean(rewardQ08, axis=0))
    np.save('rDQ1.npy', np.mean(rewardDQ1, axis=0))
    np.save('rDQ08.npy', np.mean(rewardDQ08, axis=0))
