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
from PyPi.utils.parameters import Parameter, DecayParameter


def experiment(algorithm_class, decay_exp):
    np.random.seed()

    # MDP
    p = np.load('chain_structure/p.npy')
    rew = np.load('chain_structure/rew.npy')
    mdp = FiniteMDP(p, rew, gamma=.9)

    # Policy
    epsilon = Parameter(value=1)
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
    collect_max_Q = CollectMaxQ(approximator, np.array([0]),
                                mdp.action_space.values)
    callbacks = [collect_max_Q]
    core = Core(agent, mdp, callbacks)

    # Train
    core.learn(n_iterations=5000, how_many=1, n_fit_steps=1,
               iterate_over='samples')

    _, _, reward, _, _, _ = parse_dataset(core.get_dataset())
    max_Qs = collect_max_Q.get_values()

    return reward, max_Qs

if __name__ == '__main__':
    n_experiment = 100

    logger.Logger(3)

    names = {1: '1', .51: '51', QLearning: 'Q', DoubleQLearning: 'DQ',
             WeightedQLearning: 'WQ', SpeedyQLearning: 'SPQ'}

    for e in [1, .51]:
        for a in [QLearning, DoubleQLearning, WeightedQLearning,
                  SpeedyQLearning]:
            out = Parallel(n_jobs=-1)(
                delayed(experiment)(a, e) for _ in xrange(n_experiment))
            r = np.array([o[0] for o in out])
            max_Qs = np.array([o[1] for o in out])

            r = np.mean(r, 0)
            max_Qs = np.mean(max_Qs, 0)

            np.save('r' + names[a] + names[e] + '.npy', r)
            np.save('max' + names[a] + names[e] + '.npy', max_Qs)
