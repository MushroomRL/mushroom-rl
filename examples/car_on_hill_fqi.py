import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor

from mushroom_rl.algorithms.value import FQI
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter

"""
This script aims to replicate the experiments on the Car on Hill MDP as
presented in:
"Tree-Based Batch Mode Reinforcement Learning", Ernst D. et al.. 2005. 

"""


def experiment():
    np.random.seed()

    # MDP
    mdp = CarOnHill()

    # Policy
    epsilon = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon)

    # Approximator
    approximator_params = dict(input_shape=mdp.info.observation_space.shape,
                               n_actions=mdp.info.action_space.n,
                               n_estimators=50,
                               min_samples_split=5,
                               min_samples_leaf=2)
    approximator = ExtraTreesRegressor

    # Agent
    algorithm_params = dict(n_iterations=20)
    agent = FQI(mdp.info, pi, approximator,
                approximator_params=approximator_params, **algorithm_params)

    # Algorithm
    core = Core(agent, mdp)

    # Render
    core.evaluate(n_episodes=1, render=True)

    # Train
    core.learn(n_episodes=1000, n_episodes_per_fit=1000)

    # Test
    test_epsilon = Parameter(0.)
    agent.policy.set_epsilon(test_epsilon)

    initial_states = np.zeros((289, 2))
    cont = 0
    for i in range(-8, 9):
        for j in range(-8, 9):
            initial_states[cont, :] = [0.125 * i, 0.375 * j]
            cont += 1

    dataset = core.evaluate(initial_states=initial_states)

    # Render
    core.evaluate(n_episodes=3, render=True)

    return np.mean(dataset.discounted_return)


if __name__ == '__main__':
    n_experiment = 1

    logger = Logger(FQI.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + FQI.__name__)

    Js = Parallel(n_jobs=None)(delayed(experiment)() for _ in range(n_experiment))
    logger.info((np.mean(Js)))
