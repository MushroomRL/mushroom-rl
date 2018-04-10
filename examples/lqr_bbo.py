import numpy as np

from mushroom.algorithms.policy_search import RWR, PGPE, REPS
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.core import Core
from mushroom.distributions import GaussianCholeskyDistribution
from mushroom.environments import LQR
from mushroom.policy import DeterministicPolicy
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import AdaptiveParameter

from tqdm import tqdm


"""
This script aims to replicate the experiments on the LQR MDP 
using black box optimization algorithms.

"""

tqdm.monitor_interval = 0


def experiment(alg, params, n_epochs, fit_per_run, ep_per_run):
    np.random.seed()

    # MDP
    mdp = LQR.generate(dimensions=1)

    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape)

    policy = DeterministicPolicy(mu=approximator)

    mu = np.zeros(policy.weights_size)
    sigma = 1e-3 * np.eye(policy.weights_size)
    distribution = GaussianCholeskyDistribution(mu, sigma)

    # Agent
    agent = alg(distribution, policy, mdp.info, **params)

    # Train
    core = Core(agent, mdp)
    dataset_eval = core.evaluate(n_episodes=ep_per_run)
    print('distribution parameters: ', distribution.get_parameters())
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))

    for i in range(n_epochs):
        core.learn(n_episodes=fit_per_run * ep_per_run,
                   n_episodes_per_fit=ep_per_run)
        dataset_eval = core.evaluate(n_episodes=ep_per_run)
        print('distribution parameters: ', distribution.get_parameters())
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        print('J at iteration ' + str(i) + ': ' + str(np.mean(J)))


if __name__ == '__main__':
    learning_rate = AdaptiveParameter(value=0.05)

    algs = [REPS, RWR, PGPE]
    params = [{'eps': 0.5}, {'beta': 0.7}, {'learning_rate': learning_rate}]

    for alg, params in zip(algs, params):
        print(alg.__name__)
        experiment(alg, params, n_epochs=4, fit_per_run=10, ep_per_run=100)
