import numpy as np
from joblib import Parallel, delayed

from PyPi.algorithms.td import QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning
from PyPi.approximators import Ensemble, Regressor, Tabular
from PyPi.core.core import Core
from PyPi.environments import *
from PyPi.policy import EpsGreedy
from PyPi.utils.callbacks import CollectMaxQ
from PyPi.utils import logger
from PyPi.utils.dataset import parse_dataset
from PyPi.utils.parameters import DecayParameter


def experiment(algorithm_class, decay_exp):
    np.random.seed()

    # MDP
    mdp = GridWorldVanHasselt()

    # Policy
    epsilon = DecayParameter(value=1, decay_exp=.5,
                             shape=mdp.observation_space.shape)
    pi = EpsGreedy(epsilon=epsilon, observation_space=mdp.observation_space,
                   action_space=mdp.action_space)

    # Approximator
    shape = mdp.observation_space.shape + mdp.action_space.shape
    approximator_params = dict(shape=shape)
    if algorithm_class in [QLearning, WeightedQLearning, SpeedyQLearning]:
        approximator = Regressor(Tabular, **approximator_params)
    elif algorithm_class is DoubleQLearning:
        approximator = Ensemble(Tabular, 2, **approximator_params)

    # Agent
    learning_rate = DecayParameter(value=1, decay_exp=decay_exp, shape=shape)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = algorithm_class(approximator, pi, **agent_params)

    # Algorithm
    collect_max_Q = CollectMaxQ(approximator, np.array([mdp._start]),
                                mdp.action_space.values)
    callbacks = [collect_max_Q]
    core = Core(agent, mdp, callbacks)

    # Train
    core.learn(n_iterations=10000, how_many=1, n_fit_steps=1,
               iterate_over='samples')

    _, _, reward, _, _, _ = parse_dataset(core.get_dataset())
    max_Qs = collect_max_Q.get_values()

    return reward, max_Qs

if __name__ == '__main__':
    n_experiment = 10000

    logger.Logger(3)

    names = {1: '1', .8: '08', QLearning: 'Q', DoubleQLearning: 'DQ',
             WeightedQLearning: 'WQ', SpeedyQLearning: 'SPQ'}

    for e in [1, .8]:
        for a in [QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning]:
            out = Parallel(n_jobs=-1)(
                delayed(experiment)(a, e) for _ in xrange(n_experiment))
            r = np.array([o[0] for o in out])
            max_Qs = np.array([o[1] for o in out])

            r = np.convolve(np.mean(r, 0), np.ones(100) / 100., 'valid')
            max_Qs = np.mean(max_Qs, 0)

            from matplotlib import pyplot as plt
            plt.figure()
            plt.suptitle(names[a] + ' ' + names[e])
            plt.subplot(2, 1, 1)
            plt.plot(r)
            plt.subplot(2, 1, 2)
            plt.plot(max_Qs)
    plt.show()
