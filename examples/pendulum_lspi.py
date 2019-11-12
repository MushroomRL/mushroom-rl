import numpy as np
from joblib import Parallel, delayed

from mushroom.algorithms.value import LSPI
from mushroom.core import Core
from mushroom.environments import *
from mushroom.features import Features
from mushroom.features.basis import PolynomialBasis, GaussianRBF
from mushroom.policy import EpsGreedy
from mushroom.utils.dataset import episodes_length
from mushroom.utils.parameters import Parameter


"""
This script aims to replicate the experiments on the Inverted Pendulum MDP as
presented in:
"Least-Squares Policy Iteration". Lagoudakis M. G. and Parr R.. 2003.

"""


def experiment():
    np.random.seed()

    # MDP
    mdp = CartPole()

    # Policy
    epsilon = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    basis = [PolynomialBasis()]

    s1 = np.array([-np.pi, 0, np.pi]) * .25
    s2 = np.array([-1, 0, 1])
    for i in s1:
        for j in s2:
            basis.append(GaussianRBF(np.array([i, j]), np.array([1.])))
    features = Features(basis_list=basis)

    fit_params = dict()
    approximator_params = dict(input_shape=(features.size,),
                               output_shape=(mdp.info.action_space.n,),
                               n_actions=mdp.info.action_space.n)
    agent = LSPI(pi, mdp.info, fit_params=fit_params,
                 approximator_params=approximator_params, features=features)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_episodes=100, n_episodes_per_fit=100)

    # Test
    test_epsilon = Parameter(0.)
    agent.policy.set_epsilon(test_epsilon)

    dataset = core.evaluate(n_episodes=1, quiet=True)

    return np.mean(episodes_length(dataset))


if __name__ == '__main__':
    n_experiment = 1

    steps = Parallel(n_jobs=-1)(delayed(experiment)() for _ in range(
        n_experiment))
    print(np.mean(steps))
