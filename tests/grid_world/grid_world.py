import numpy as np
from joblib import Parallel, delayed

from mushroom.algorithms.td import QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning
from mushroom.core.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.callbacks import CollectDataset, CollectMaxQ
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.parameters import DecayParameter


def experiment(algorithm_class):
    np.random.seed(20)

    # MDP
    mdp = GridWorldVanHasselt()

    # Policy
    epsilon = DecayParameter(value=1, decay_exp=.5,
                             shape=mdp.observation_space.size)
    pi = EpsGreedy(epsilon=epsilon, observation_space=mdp.observation_space,
                   action_space=mdp.action_space)

    # Agent
    shape = mdp.observation_space.size + mdp.action_space.size
    learning_rate = DecayParameter(value=1., decay_exp=1., shape=shape)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = algorithm_class(shape, pi, mdp.gamma, agent_params)

    # Algorithm
    collect_dataset = CollectDataset()
    collect_max_Q = CollectMaxQ(agent.approximator, np.array([mdp._start]))
    callbacks = [collect_dataset, collect_max_Q]
    core = Core(agent, mdp, callbacks)

    # Train
    core.learn(n_iterations=2000, how_many=1, n_fit_steps=1,
               iterate_over='samples', quiet=True)

    _, _, reward, _, _, _ = parse_dataset(collect_dataset.get())
    max_Qs = collect_max_Q.get_values()

    return reward, max_Qs


if __name__ == '__main__':
    print('Executing grid_world test...')

    n_experiment = 2

    names = ['Q', 'DQ', 'WQ', 'SQ']
    for i, a in enumerate([QLearning, DoubleQLearning, WeightedQLearning,
                           SpeedyQLearning]):
        out = Parallel(n_jobs=-1)(
            delayed(experiment)(a) for _ in xrange(n_experiment))
        r = np.array([o[0] for o in out])
        max_Qs = np.array([o[1] for o in out])

        r = np.convolve(np.mean(r, 0), np.ones(100) / 100., 'valid')
        max_Qs = np.mean(max_Qs, 0)

        assert np.array_equal(
            np.load('tests/grid_world/r' + names[i] + '.npy'), r)
        assert np.array_equal(
            np.load('tests/grid_world/max' + names[i] + '.npy'), max_Qs)
