import numpy as np

from mushroom.algorithms.value import SARSA
from mushroom.core import Core
from mushroom.environments.generators.taxi import generate_taxi
from mushroom.policy import Boltzmann, EpsGreedy, Mellowmax
from mushroom.utils.callbacks import CollectDataset
from mushroom.utils.parameters import Parameter


def experiment(policy, value):
    np.random.seed(45)

    # MDP
    mdp = generate_taxi('tests/taxi/grid.txt', rew=(0, 1, 5))

    # Policy
    pi = policy(Parameter(value=value))

    # Agent
    learning_rate = Parameter(value=.15)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = SARSA(pi, mdp.info, agent_params)

    # Algorithm
    collect_dataset = CollectDataset()
    callbacks = [collect_dataset]
    core = Core(agent, mdp, callbacks)

    # Train
    n_steps = 2000
    core.learn(n_steps=n_steps, n_steps_per_fit=1, quiet=True)

    return np.sum(np.array(collect_dataset.get())[:, 2]) / float(n_steps)


if __name__ == '__main__':
    print('Executing taxi test...')

    n_experiment = 1

    algs = {EpsGreedy: 'epsilon', Boltzmann: 'boltzmann', Mellowmax: 'mellow'}

    Js = list()
    for p in [EpsGreedy, Boltzmann, Mellowmax]:
        Js.append(np.mean(experiment(p, 1.)))

    assert np.round(Js[0], 4) == .016
    assert np.round(Js[1], 4) == .0285
    assert np.round(Js[2], 4) == .0205
