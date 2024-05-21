import numpy as np
from joblib import Parallel, delayed
from pathlib import Path

from mushroom_rl.algorithms.value import QLearning, DoubleQLearning,\
    WeightedQLearning, SpeedyQLearning
from mushroom_rl.core import Core
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.callbacks import CollectQ
from mushroom_rl.rl_utils.parameters import Parameter, DecayParameter


"""
Simple script to solve a double chain with Q-Learning and some of its variants.
The considered double chain is the one presented in:
"Relative Entropy Policy Search". Peters J. et al.. 2010.

"""


def experiment(algorithm_class, exp):
    np.random.seed()

    # MDP
    path = Path(__file__).resolve().parent / 'chain_structure'
    p = np.load(path / 'p.npy')
    rew = np.load(path / 'rew.npy')
    mdp = FiniteMDP(p, rew, gamma=.9)

    # Policy
    epsilon = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = DecayParameter(value=1., exp=exp, size=mdp.info.size)
    algorithm_params = dict(learning_rate=learning_rate)
    agent = algorithm_class(mdp.info, pi, **algorithm_params)

    # Algorithm
    collect_Q = CollectQ(agent.Q)
    callbacks = [collect_Q]
    core = Core(agent, mdp, callbacks)

    # Train
    core.learn(n_steps=20000, n_steps_per_fit=1, quiet=True)

    Qs = collect_Q.get()

    return Qs


if __name__ == '__main__':
    n_experiment = 5

    names = {1: '1', .51: '51', QLearning: 'Q', DoubleQLearning: 'DQ',
             WeightedQLearning: 'WQ', SpeedyQLearning: 'SPQ'}

    log_path = Path(__file__).resolve().parent / 'logs'

    log_path.mkdir(parents=True, exist_ok=True)

    for e in [1, .51]:
        for a in [QLearning, DoubleQLearning, WeightedQLearning,
                  SpeedyQLearning]:
            out = Parallel(n_jobs=1)(
                delayed(experiment)(a, e) for _ in range(n_experiment))
            Qs = np.array([o for o in out])

            Qs = np.mean(Qs, 0)

            filename = names[a] + names[e] + '.npy'
            np.save(log_path / filename, Qs[:, 0, 0])
