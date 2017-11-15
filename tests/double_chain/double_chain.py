import numpy as np
from joblib import Parallel, delayed

from mushroom.algorithms.value.td import QLearning, DoubleQLearning,\
    WeightedQLearning, SpeedyQLearning
from mushroom.core.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.callbacks import CollectQ
from mushroom.utils.parameters import Parameter, ExponentialDecayParameter


def experiment(algorithm_class, decay_exp):
    np.random.seed(3)

    # MDP
    p = np.load('tests/double_chain/chain_structure/p.npy')
    rew = np.load('tests/double_chain/chain_structure/rew.npy')
    mdp = FiniteMDP(p, rew, gamma=.9)

    # Policy
    epsilon = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = ExponentialDecayParameter(value=1., decay_exp=decay_exp,
                                              size=mdp.info.size)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = algorithm_class(pi, mdp.info, agent_params)

    # Algorithm
    collect_Q = CollectQ(agent.approximator)
    callbacks = [collect_Q]
    core = Core(agent, mdp, callbacks)

    # Train
    core.learn(n_iterations=2000, how_many=1, n_fit_steps=1,
               iterate_over='samples', quiet=True)

    Qs = collect_Q.get_values()

    return Qs


if __name__ == '__main__':
    print('Executing double_chain test...')

    n_experiment = 1

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
