import numpy as np
from joblib import Parallel, delayed

from mushroom_rl.algorithms.value import SARSA
from mushroom_rl.core import Core
from mushroom_rl.environments.generators.taxi import generate_taxi
from mushroom_rl.policy import Boltzmann, EpsGreedy, Mellowmax
from mushroom_rl.utils.callbacks import CollectDataset
from mushroom_rl.rl_utils.parameters import Parameter


"""
This script aims to replicate the experiments on the Taxi MDP as presented in:
"An Alternative Softmax Operator for Reinforcement Learning", Asadi K. et al..
2017. 

"""


def experiment(policy, value):
    np.random.seed()

    # MDP
    mdp = generate_taxi('grid.txt')

    # Policy
    pi = policy(Parameter(value=value))

    # Agent
    learning_rate = Parameter(value=.15)
    algorithm_params = dict(learning_rate=learning_rate)
    agent = SARSA(mdp.info, pi, **algorithm_params)

    # Algorithm
    collect_dataset = CollectDataset()
    callbacks = [collect_dataset]
    core = Core(agent, mdp, callbacks)

    # Train
    n_steps = 300000
    core.learn(n_steps=n_steps, n_steps_per_fit=1, quiet=True)

    return np.sum(np.array(collect_dataset.get())[:, 2]) / float(n_steps)


if __name__ == '__main__':
    n_experiment = 25

    algs = {EpsGreedy: 'epsilon', Boltzmann: 'boltzmann', Mellowmax: 'mellow'}
    ranges = {EpsGreedy: np.linspace(.05, .5, 10),
              Boltzmann: np.linspace(.5, 10, 10),
              Mellowmax: np.linspace(.5, 10, 10)}

    for p in [EpsGreedy, Boltzmann, Mellowmax]:
        print('Policy: ', algs[p])
        Js = list()
        for v in ranges[p]:
            out = Parallel(n_jobs=-1)(
                delayed(experiment)(p, v) for _ in range(n_experiment))
            J = [np.mean(o) for o in out]
            Js.append(np.mean(J))

        np.save('r_%s.npy' % algs[p], Js)
