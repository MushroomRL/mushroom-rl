import gym
import numpy as np
from joblib import Parallel, delayed

from mushroom.algorithms.value import TrueOnlineSARSALambda
from mushroom.core import Core
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
    gym.logger.setLevel(0)
    np.random.seed(386)

    # MDP
    mdp = Gym(name='MountainCar-v0', horizon=10000, gamma=1.)
    mdp.seed(201)

    # Policy
    epsilon = Parameter(value=0.)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = Parameter(alpha)
    tilings = Tiles.generate(5, [5, 5],
                             mdp.info.observation_space.low,
                             mdp.info.observation_space.high)
    features = Features(tilings=tilings)

    approximator_params = dict(input_shape=(features.size,),
                               output_shape=(mdp.info.action_space.n,),
                               n_actions=mdp.info.action_space.n)
    algorithm_params = {'learning_rate': learning_rate,
                        'lambda_coeff': .9}
    agent = TrueOnlineSARSALambda(pi, mdp.info,
                                  approximator_params=approximator_params,
                                  features=features, **algorithm_params)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_episodes=20, n_steps_per_fit=1, quiet=True)

    # Test
    test_epsilon = Parameter(0.)
    agent.policy.set_epsilon(test_epsilon)

    initial_states = np.array([[0., 0.], [.1, .1]])
    dataset = core.evaluate(initial_states=initial_states, quiet=True)

    return np.mean(compute_J(dataset, 1.))


if __name__ == '__main__':
    print('Executing mountain_car test...')

    alpha = .1
    J = experiment(alpha)

    assert J == -135.5
