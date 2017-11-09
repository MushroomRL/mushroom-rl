import numpy as np
from joblib import Parallel, delayed

from mushroom.algorithms.value.td import TrueOnlineSARSALambda
from mushroom.core.core import Core
from mushroom.environments import *
from mushroom.features import Features
from mushroom.features.tiles import Tiles
from mushroom.policy import EpsGreedy
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import Parameter

"""
This script uses True Online Sarsa(lambda) to replicate experiments in Seijen
H. V. et al. (2014).

"""


def experiment(alpha):
    np.random.seed(386)

    # MDP
    mdp = Gym(name='MountainCar-v0', horizon=10000, gamma=1.)
    mdp.seed(3)

    # Policy
    epsilon = Parameter(value=0.)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = Parameter(alpha)
    tilings = list()
    n_tilings = 10
    tiles_dim = 10
    offset = 1. / (n_tilings * tiles_dim)
    for i in xrange(10):
        x_min = mdp.info.observation_space.low - (n_tilings - 1 - i) * offset
        x_max = mdp.info.observation_space.high + i * offset
        x_range = [[x, y] for x, y in zip(x_min, x_max)]
        tilings.append(Tiles(x_range, [n_tilings, n_tilings]))
    features = Features(tilings=tilings)

    approximator_params = dict(input_shape=(features.size,),
                               output_shape=(mdp.info.action_space.n,),
                               n_actions=mdp.info.action_space.n)
    algorithm_params = {'learning_rate': learning_rate,
                        'lambda': .9}
    fit_params = dict()
    agent_params = {'approximator_params': approximator_params,
                    'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = TrueOnlineSARSALambda(pi, mdp.info, agent_params, features)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_iterations=2000, how_many=1, n_fit_steps=1,
               iterate_over='samples')
    core.reset()

    # Test
    test_epsilon = Parameter(0.)
    agent.policy.set_epsilon(test_epsilon)

    initial_states = np.array([[0., 0.], [.1, .1]])
    dataset = core.evaluate(initial_states=initial_states)

    return np.mean(compute_J(dataset, 1.))


if __name__ == '__main__':
    n_experiment = 2

    alpha = .1
    Js = Parallel(
        n_jobs=-1)(delayed(experiment)(alpha) for _ in range(n_experiment))

    assert np.mean(Js) == -399.5
