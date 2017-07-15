import numpy as np
from joblib import Parallel, delayed

from keras.models import Sequential

from PyPi.algorithms.dqn import DQN
from PyPi.approximators import Regressor
from PyPi.core.core import Core
from PyPi.environments import *
from PyPi.policy import EpsGreedy
from PyPi.utils import logger
from PyPi.utils.parameters import Parameter


def experiment():
    np.random.seed()

    # MDP
    mdp = Atari('BreakoutDeterministic-v3')

    # Policy
    epsilon = Parameter(value=1)
    pi = EpsGreedy(epsilon=epsilon, observation_space=mdp.observation_space,
                   action_space=mdp.action_space)

    # Approximator
    approximator_params = dict()
    nn = Sequential()
    approximator = Regressor(nn, **approximator_params)

    # Agent
    algorithm_params = dict()
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = DQN(approximator, pi, **agent_params)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_iterations=np.inf, how_many=32, n_fit_steps=1,
               iterate_over='samples')


if __name__ == '__main__':
    n_experiment = 1

    logger.Logger(3)

    Parallel(n_jobs=-1)(delayed(experiment)() for _ in range(n_experiment))
