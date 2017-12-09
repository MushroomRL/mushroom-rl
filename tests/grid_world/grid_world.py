import numpy as np
from joblib import Parallel, delayed

from mushroom.algorithms.value import QLearning, DoubleQLearning,\
    WeightedQLearning, SpeedyQLearning, SARSA
from mushroom.core.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.callbacks import CollectDataset, CollectMaxQ
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.parameters import ExponentialDecayParameter


def experiment(algorithm_class):
    np.random.seed(20)

    # MDP
    mdp = GridWorldVanHasselt()

    # Policy
    epsilon = ExponentialDecayParameter(value=1, decay_exp=.5,
                                        size=mdp.info.observation_space.size)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = ExponentialDecayParameter(value=1., decay_exp=1.,
                                              size=mdp.info.size)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = algorithm_class(pi, mdp.info, agent_params)

    # Algorithm
    collect_dataset = CollectDataset()
    start = mdp.convert_to_int(mdp._start, mdp._width)
    collect_max_Q = CollectMaxQ(agent.approximator, start)
    callbacks = [collect_dataset, collect_max_Q]
    core = Core(agent, mdp, callbacks)

    # Train
    core.learn(n_steps=2000, n_steps_per_fit=1, quiet=True)

    _, _, reward, _, _, _ = parse_dataset(collect_dataset.get())
    max_Qs = collect_max_Q.get_values()

    return reward, max_Qs


if __name__ == '__main__':
    print('Executing grid_world test...')

    n_experiment = 1

    names = ['Q', 'DQ', 'WQ', 'SQ', 'SARSA']
    for i, a in enumerate([QLearning, DoubleQLearning, WeightedQLearning,
                           SpeedyQLearning, SARSA]):
        out = Parallel(n_jobs=-1)(
            delayed(experiment)(a) for _ in xrange(n_experiment))
        r = np.array([o[0] for o in out])
        max_Qs = np.array([o[1] for o in out])

        r = np.convolve(np.mean(r, 0), np.ones(100) / 100., 'valid')
        r_old = np.load('tests/grid_world/r' + names[i] + '.npy')
        max_Qs = np.mean(max_Qs, 0)
        max_q_old = np.load('tests/grid_world/max' + names[i] + '.npy')

        assert np.array_equal(r_old, r)
        assert np.array_equal(max_q_old, max_Qs)
