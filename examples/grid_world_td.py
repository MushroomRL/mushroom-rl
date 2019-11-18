import matplotlib
matplotlib.use('Agg')

import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

from mushroom.algorithms.value import QLearning, DoubleQLearning,\
    WeightedQLearning, SpeedyQLearning, SARSA
from mushroom.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.callbacks import CollectDataset, CollectMaxQ
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.parameters import ExponentialParameter


"""
This script aims to replicate the experiments on the Grid World MDP as
presented in:
"Double Q-Learning", Hasselt H. V.. 2010.

SARSA and many variants of Q-Learning are used. 

"""


def experiment(algorithm_class, exp):
    np.random.seed()

    # MDP
    mdp = GridWorldVanHasselt()

    # Policy
    epsilon = ExponentialParameter(value=1, exp=.5, size=mdp.info.observation_space.size)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = ExponentialParameter(value=1, exp=exp, size=mdp.info.size)
    algorithm_params = dict(learning_rate=learning_rate)
    agent = algorithm_class(pi, mdp.info, **algorithm_params)

    # Algorithm
    start = mdp.convert_to_int(mdp._start, mdp._width)
    collect_max_Q = CollectMaxQ(agent.approximator, start)
    collect_dataset = CollectDataset()
    callbacks = [collect_dataset, collect_max_Q]
    core = Core(agent, mdp, callbacks)

    # Train
    core.learn(n_steps=10000, n_steps_per_fit=1, quiet=True)

    _, _, reward, _, _, _ = parse_dataset(collect_dataset.get())
    max_Qs = collect_max_Q.get()

    return reward, max_Qs


if __name__ == '__main__':
    n_experiment = 10000

    names = {1: '1', .8: '08', QLearning: 'Q', DoubleQLearning: 'DQ',
             WeightedQLearning: 'WQ', SpeedyQLearning: 'SPQ', SARSA: 'SARSA'}

    for e in [1, .8]:
        print('Exp: ', e)
        fig = plt.figure()
        plt.suptitle(names[e])
        legend_labels = []
        for a in [QLearning, DoubleQLearning, WeightedQLearning,
                  SpeedyQLearning, SARSA]:
            print('Alg: ', names[a])
            out = Parallel(n_jobs=-1)(
                delayed(experiment)(a, e) for _ in range(n_experiment))
            r = np.array([o[0] for o in out])
            max_Qs = np.array([o[1] for o in out])

            r = np.convolve(np.mean(r, 0), np.ones(100) / 100., 'valid')
            max_Qs = np.mean(max_Qs, 0)

            np.save(names[a] + '_' + names[e] + '_r.npy', r)
            np.save(names[a] + '_' + names[e] + '_maxQ.npy', max_Qs)

            plt.subplot(2, 1, 1)
            plt.plot(r)
            plt.subplot(2, 1, 2)
            plt.plot(max_Qs)
            legend_labels.append(names[a])
        plt.legend(legend_labels)
        fig.savefig('test_' + names[e] + '.png')
