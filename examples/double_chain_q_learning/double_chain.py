import numpy as np
from joblib import Parallel, delayed

from mushroom.algorithms.value import QLearning, DoubleQLearning,\
    WeightedQLearning, SpeedyQLearning
from mushroom.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.callbacks import CollectQ
from mushroom.utils.parameters import Parameter, ExponentialParameter


"""
Simple script to solve a double chain with Q-Learning and some of its variants.
The considered double chain is the one presented in:
"Relative Entropy Policy Search". Peters J. et al.. 2010.

"""


def experiment(algorithm_class, exp):
    np.random.seed()

    # MDP
    p = np.load('chain_structure/p.npy')
    rew = np.load('chain_structure/rew.npy')
    mdp = FiniteMDP(p, rew, gamma=.9)

    # Policy
    epsilon = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = ExponentialParameter(value=1., exp=exp, size=mdp.info.size)
    algorithm_params = dict(learning_rate=learning_rate)
    agent = algorithm_class(mdp.info, pi, **algorithm_params)

    # Algorithm
    collect_Q = CollectQ(agent.approximator)
    callbacks = [collect_Q]
    core = Core(agent, mdp, callbacks)

    # Train
    core.learn(n_steps=20000, n_steps_per_fit=1, quiet=True)

    Qs = collect_Q.get()

    return Qs


if __name__ == '__main__':
    n_experiment = 500

    names = {1: '1', .51: '51', QLearning: 'Q', DoubleQLearning: 'DQ',
             WeightedQLearning: 'WQ', SpeedyQLearning: 'SPQ'}

    for e in [1, .51]:
        for a in [QLearning, DoubleQLearning, WeightedQLearning,
                  SpeedyQLearning]:
            out = Parallel(n_jobs=-1)(
                delayed(experiment)(a, e) for _ in range(n_experiment))
            Qs = np.array([o for o in out])

            Qs = np.mean(Qs, 0)

            np.save(names[a] + names[e] + '.npy', Qs[:, 0, 0])
