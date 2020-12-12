import numpy as np

from mushroom_rl.algorithms.policy_search import REINFORCE, GPOMDP, eNAC
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.core import Core
from mushroom_rl.environments import LQR
from mushroom_rl.policy import StateStdGaussianPolicy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.optimizers import AdaptiveOptimizer

from tqdm import tqdm


"""
This script aims to replicate the experiments on the LQR MDP using policy
gradient algorithms.

"""

tqdm.monitor_interval = 0


def experiment(alg, n_epochs, n_iterations, ep_per_run):
    np.random.seed()

    # MDP
    mdp = LQR.generate(dimensions=1)

    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape)

    sigma = Regressor(LinearApproximator,
                      input_shape=mdp.info.observation_space.shape,
                      output_shape=mdp.info.action_space.shape)

    sigma_weights = 2 * np.ones(sigma.weights_size)
    sigma.set_weights(sigma_weights)

    policy = StateStdGaussianPolicy(approximator, sigma)

    # Agent
    optimizer = AdaptiveOptimizer(eps=.01)
    algorithm_params = dict(optimizer=optimizer)
    agent = alg(mdp.info, policy, **algorithm_params)

    # Train
    core = Core(agent, mdp)
    dataset_eval = core.evaluate(n_episodes=ep_per_run)
    print('policy parameters: ', policy.get_weights())
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))

    for i in range(n_epochs):
        core.learn(n_episodes=n_iterations * ep_per_run,
                   n_episodes_per_fit=ep_per_run)
        dataset_eval = core.evaluate(n_episodes=ep_per_run)
        print('policy parameters: ', policy.get_weights())
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        print('J at iteration ' + str(i) + ': ' + str(np.mean(J)))


if __name__ == '__main__':

    algs = [REINFORCE, GPOMDP, eNAC]

    for alg in algs:
        print(alg.__name__)
        experiment(alg, n_epochs=10, n_iterations=4, ep_per_run=100)
