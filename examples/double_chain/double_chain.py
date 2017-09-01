import numpy as np
from joblib import Parallel, delayed

from mushroom.algorithms.td import QLearning, DoubleQLearning, WeightedQLearning,\
    SpeedyQLearning
from mushroom.approximators import Ensemble, Regressor, Tabular
from mushroom.core.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.callbacks import CollectQ
from mushroom.utils.parameters import Parameter, DecayParameter


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
    agent = algorithm_class(approximator, pi, mdp.gamma, **agent_params)

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
    n_experiment = 500

    names = {1: '1', .51: '51', QLearning: 'Q', DoubleQLearning: 'DQ',
             WeightedQLearning: 'WQ', SpeedyQLearning: 'SPQ'}

    for e in [1, .51]:
        for a in [QLearning, DoubleQLearning, WeightedQLearning,
                  SpeedyQLearning]:
            out = Parallel(n_jobs=-1)(
                delayed(experiment)(a, e) for _ in xrange(n_experiment))
            Qs = np.array([o for o in out])

            Qs = np.mean(Qs, 0)

            np.save(names[a] + names[e] + '.npy', Qs[:, 0, 0])
