import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np
from joblib import Parallel, delayed

from mushroom.algorithms.td import QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning
from mushroom.core.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.callbacks import CollectDataset, CollectMaxQ
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.parameters import DecayParameter



def experiment(algorithm_class, decay_exp):
    np.random.seed()

    # MDP
    mdp = GridWorldVanHasselt()

    # Policy
    epsilon = DecayParameter(value=1, decay_exp=.5,
                             shape=mdp.observation_space.size)
    pi = EpsGreedy(epsilon=epsilon, observation_space=mdp.observation_space,
                   action_space=mdp.action_space)

    # Agent
    shape = mdp.observation_space.size + mdp.action_space.size
    learning_rate = DecayParameter(value=1, decay_exp=decay_exp, shape=shape)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = algorithm_class(shape, pi, mdp.gamma, **agent_params)

    # Algorithm
    collect_max_Q = CollectMaxQ(agent.approximator, np.array([mdp._start]))
    collect_dataset = CollectDataset()
    callbacks = [collect_dataset, collect_max_Q]
    core = Core(agent, mdp, callbacks)

    # Train
    core.learn(n_iterations=10000, how_many=1, n_fit_steps=1,
               iterate_over='samples', quiet=True)

    _, _, reward, _, _, _ = parse_dataset(collect_dataset.get())
    max_Qs = collect_max_Q.get_values()

    return reward, max_Qs


if __name__ == '__main__':
    n_experiment = 10000

    names = {1: '1', .8: '08', QLearning: 'Q', DoubleQLearning: 'DQ',
             WeightedQLearning: 'WQ', SpeedyQLearning: 'SPQ'}

    for e in [1, .8]:
        fig = plt.figure()
        plt.suptitle(names[e])
        legend_labels = []
        for a in [QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning]:
            print names[a]
            out = Parallel(n_jobs=-1)(
                delayed(experiment)(a, e) for _ in xrange(n_experiment))
            r = np.array([o[0] for o in out])
            max_Qs = np.array([o[1] for o in out])

            r = np.convolve(np.mean(r, 0), np.ones(100) / 100., 'valid')
            max_Qs = np.mean(max_Qs, 0)

            plt.subplot(2, 1, 1)
            plt.plot(r)
            plt.subplot(2, 1, 2)
            plt.plot(max_Qs)
            legend_labels.append(names[a])
        plt.legend(legend_labels)
        fig.savefig('test_' + names[e] + '.png')
