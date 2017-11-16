import numpy as np
from joblib import Parallel, delayed

from mushroom.algorithms.value.batch_td import LSPI
from mushroom.approximators.parametric import LinearApproximator
from mushroom.core.core import Core
from mushroom.environments import *
from mushroom.features import Features
from mushroom.features.tiles import Tiles
from mushroom.policy import EpsGreedy
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import Parameter

"""
This script uses Least-Squares Policy Iteration to replicate experiments in
Lagoudakis M. G. and Parr R. (2003).

"""


def experiment():
    np.random.seed()

    # MDP
    mdp = InvertedPendulum()

    # Policy
    epsilon = Parameter(value=0.)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    tilings = Tiles.generate(10, [10, 10],
                             mdp.info.observation_space.low,
                             mdp.info.observation_space.high)
    features = Features(tilings=tilings)

    approximator_params = dict(input_shape=(features.size,),
                               output_shape=(mdp.info.action_space.n,),
                               n_actions=mdp.info.action_space.n)
    algorithm_params = dict()
    fit_params = dict()
    agent_params = {'approximator_params': approximator_params,
                    'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = LSPI(pi, mdp.info, agent_params, features)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_episodes=20, n_episodes_per_fit=20)

    # Test
    test_epsilon = Parameter(0.)
    agent.policy.set_epsilon(test_epsilon)

    dataset = core.evaluate(n_episodes=20)

    return np.mean(compute_J(dataset, 1.))


if __name__ == '__main__':
    n_experiment = 1

    Js = Parallel(
        n_jobs=-1)(delayed(experiment)() for _ in range(n_experiment))

    print(np.mean(Js))
