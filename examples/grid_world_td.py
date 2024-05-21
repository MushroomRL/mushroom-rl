import matplotlib
matplotlib.use('Agg')

import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

from mushroom_rl.algorithms.value import QLearning, DoubleQLearning,\
    WeightedQLearning, SpeedyQLearning, SARSA
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.callbacks import CollectDataset, CollectMaxQ
from mushroom_rl.rl_utils.parameters import DecayParameter


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
    epsilon = DecayParameter(value=1, exp=.5, size=mdp.info.observation_space.size)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = DecayParameter(value=1, exp=exp, size=mdp.info.size)
    algorithm_params = dict(learning_rate=learning_rate)
    agent = algorithm_class(mdp.info, pi, **algorithm_params)

    # Algorithm
    start = mdp.convert_to_int(mdp._start, mdp._width)
    collect_max_Q = CollectMaxQ(agent.Q, start)
    collect_dataset = CollectDataset()
    callbacks = [collect_dataset, collect_max_Q]
    core = Core(agent, mdp, callbacks)

    # Train
    core.learn(n_steps=10000, n_steps_per_fit=1, quiet=True)

    reward = collect_dataset.get().rewards
    max_Qs = collect_max_Q.get()

    return reward, max_Qs


if __name__ == '__main__':
    n_experiment = 10000

    logger = Logger(QLearning.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + QLearning.__name__)

    names = {1: '1', .8: '08', QLearning: 'Q', DoubleQLearning: 'DQ',
             WeightedQLearning: 'WQ', SpeedyQLearning: 'SPQ', SARSA: 'SARSA'}

    for e in [1, .8]:
        logger.info(f'Exp: {e}')
        fig = plt.figure()
        plt.suptitle(names[e])
        legend_labels = []
        for a in [QLearning, DoubleQLearning, WeightedQLearning,
                  SpeedyQLearning, SARSA]:
            logger.info(f'Alg: {names[a]}')
            out = Parallel(n_jobs=-1)(delayed(experiment)(a, e) for _ in range(n_experiment))
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
