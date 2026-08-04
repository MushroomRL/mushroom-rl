"""
This script aims to replicate the experiments on the Car on Hill MDP as presented in:
"Tree-Based Batch Mode Reinforcement Learning", Ernst D. et al. 2005.

"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from mushroom_rl.algorithms.value import FQI
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import CarOnHill
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter


def experiment(seed=0):
    np.random.seed(seed)

    # MDP
    mdp = CarOnHill()

    logger = Logger(FQI.name(), results_dir=None)
    logger.log_experiment_info(FQI, mdp)

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
    agent = FQI(mdp.info, pi, approximator, approximator_params=approximator_params, n_iterations=20)

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    # Show the environment, driven by the random policy the dataset is collected with
    core.evaluate(n_episodes=1, render=True)

    # Train
    core.learn(n_episodes=1000, n_episodes_per_fit=1000)

    # Test on the grid of initial states used by the paper
    initial_states = np.zeros((289, 2))
    cont = 0
    for i in range(-8, 9):
        for j in range(-8, 9):
            initial_states[cont, :] = [0.125 * i, 0.375 * j]
            cont += 1

    dataset = core.evaluate(initial_states=initial_states, greedy=True)
    logger.info(f'J: {dataset.discounted_return.mean()}')

    # Visualize the final policy
    core.evaluate(n_episodes=3, render=True, greedy=True)


if __name__ == '__main__':
    experiment()
