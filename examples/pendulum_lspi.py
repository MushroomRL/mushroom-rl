import numpy as np
from joblib import Parallel, delayed

from mushroom.algorithms.value import LSPI
from mushroom.core import Core
from mushroom.environments import *
from mushroom.features import Features
from mushroom.features.basis import PolynomialBasis, GaussianRBF
from mushroom.policy import EpsGreedy
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import Parameter


"""
This script aims to replicate the experiments on the Inverted Pendulum MDP as
presented in:
"Least-Squares Policy Iteration". Lagoudakis M. G. and Parr R.. 2003.

"""


def experiment():
    np.random.seed()

    # MDP
    mdp = InvertedPendulumDiscrete()

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

    approximator_params = dict(input_shape=(features.size,),
                               output_shape=(mdp.info.action_space.n,),
                               n_actions=mdp.info.action_space.n)
    algorithm_params = dict()
    agent = LSPI(pi, mdp.info, approximator_params=approximator_params,
                 features=features, **algorithm_params)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_episodes=1000, n_episodes_per_fit=1000)

    # Test
    test_epsilon = Parameter(0.)
    agent.policy.set_epsilon(test_epsilon)

    dataset = core.evaluate(n_episodes=1000)

    return np.mean(compute_J(dataset, mdp.info.gamma))


if __name__ == '__main__':
    n_experiment = 100

    Js = Parallel(n_jobs=-1)(delayed(experiment)() for _ in range(n_experiment))
    print((np.mean(Js)))
