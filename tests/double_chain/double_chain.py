import numpy as np
from joblib import Parallel, delayed

from PyPi.algorithms.td import QLearning, DoubleQLearning, WeightedQLearning,\
    SpeedyQLearning
from PyPi.approximators import Ensemble, Regressor, Tabular
from PyPi.core.core import Core
from PyPi.environments import *
from PyPi.policy import EpsGreedy
from PyPi.utils.callbacks import CollectQ
from PyPi.utils.parameters import Parameter, DecayParameter


def experiment(algorithm_class, decay_exp):
    np.random.seed(3)

    # MDP
    p = np.load('tests/double_chain/chain_structure/p.npy')
    rew = np.load('tests/double_chain/chain_structure/rew.npy')
    mdp = FiniteMDP(p, rew, gamma=.9)

    # Policy
    epsilon = Parameter(value=1)
    pi = EpsGreedy(epsilon=epsilon, observation_space=mdp.observation_space,
                   action_space=mdp.action_space)

    # Approximator
    shape = mdp.observation_space.size + mdp.action_space.size
    approximator_params = dict(shape=shape)
    if algorithm_class in [QLearning, WeightedQLearning, SpeedyQLearning]:
        approximator = Regressor(Tabular,
                                 discrete_actions=mdp.action_space.n,
                                 **approximator_params)
    elif algorithm_class is DoubleQLearning:
        approximator = Ensemble(Tabular,
                                n_models=2,
                                discrete_actions=mdp.action_space.n,
                                **approximator_params)

    # Agent
    learning_rate = DecayParameter(value=1, decay_exp=decay_exp, shape=shape)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = algorithm_class(approximator, pi, **agent_params)

    # Algorithm
    collect_Q = CollectQ(approximator)
    callbacks = [collect_Q]
    core = Core(agent, mdp, callbacks)

    # Train
    core.learn(n_iterations=20000, how_many=1, n_fit_steps=1,
               iterate_over='samples', quiet=True)

    Qs = collect_Q.get_values()

    return Qs

if __name__ == '__main__':
    print('Executing double_chain test...')

    n_experiment = 2

    names = {1: '1', .51: '51', QLearning: 'Q', DoubleQLearning: 'DQ',
             WeightedQLearning: 'WQ', SpeedyQLearning: 'SPQ'}

    for e in [1, .51]:
        for a in [QLearning, DoubleQLearning, WeightedQLearning,
                  SpeedyQLearning]:
            out = Parallel(n_jobs=-1)(
                delayed(experiment)(a, e) for _ in xrange(n_experiment))
            Qs = np.array([o for o in out])

            Qs = np.mean(Qs, 0)

            assert np.array_equal(Qs[:, 0, 0], np.load(
                'tests/double_chain/' + names[a] + names[e] + '.npy'))
