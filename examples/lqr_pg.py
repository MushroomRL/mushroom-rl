import numpy as np

from mushroom.algorithms.policy_search import REINFORCE, GPOMDP, eNAC
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.core import Core
from mushroom.environments import LQR
from mushroom.policy import MultivariateGaussianPolicy
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import AdaptiveParameter

from tqdm import tqdm


"""
This script aims to replicate the experiments on the LQR MDP 
using policy gradient algorithms.

"""

tqdm.monitor_interval = 0

def experiment(alg, n_runs, n_iterations, ep_per_run):
    np.random.seed()

    # MDP
    mdp = LQR.generate(dimensions=1)

    approximator_params = dict(input_dim=mdp.info.observation_space.shape)
    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape,
                             params=approximator_params)

    sigma = .1 * np.eye(1)
    policy = MultivariateGaussianPolicy(mu=approximator, sigma=sigma)

    # Agent
    learning_rate = AdaptiveParameter(value=.01)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = alg(policy, mdp.info, agent_params)

    # Train
    core = Core(agent, mdp)
    dataset_eval = core.evaluate(n_episodes=ep_per_run)
    print 'policy parameters: ', policy.get_weights()
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))

    for i in xrange(n_runs):
        core.learn(n_episodes=n_iterations * ep_per_run,
                   n_episodes_per_fit=ep_per_run)
        dataset_eval = core.evaluate(n_episodes=ep_per_run)
        print 'policy parameters: ', policy.get_weights()
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        print('J at iteration ' + str(i) + ': ' + str(np.mean(J)))

    np.save('ship_steering.npy', dataset_eval)


if __name__ == '__main__':

    algs = [REINFORCE, GPOMDP, eNAC]

    for alg in algs:
        print alg.__name__
        experiment(alg, n_runs=10, n_iterations=40, ep_per_run=100)
