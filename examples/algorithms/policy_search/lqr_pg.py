"""
This script shows how to solve the LQR problem with step-based policy gradient algorithms.

"""
import numpy as np

from tqdm import trange

from mushroom_rl.algorithms.policy_search import REINFORCE, GPOMDP, eNAC
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import LQR
from mushroom_rl.policy import StateStdGaussianPolicy
from mushroom_rl.rl_utils.optimizers import AdaptiveOptimizer


def experiment(alg, n_epochs, n_iterations, ep_per_run, seed=None):
    np.random.seed(seed)

    # MDP
    mdp = LQR.generate(dimensions=2, max_action=1., max_pos=1.)

    logger = Logger(alg.name(), results_dir=None)
    logger.log_experiment_info(alg, mdp, n_epochs=n_epochs, n_iterations=n_iterations,
                               ep_per_run=ep_per_run)

    # Policy
    approximator = LinearApproximator(input_shape=mdp.info.observation_space.shape,
                                      output_shape=mdp.info.action_space.shape)

    sigma = LinearApproximator(input_shape=mdp.info.observation_space.shape,
                               output_shape=mdp.info.action_space.shape)

    sigma_weights = 0.25 * np.ones(sigma.weights_size)
    sigma.set_weights(sigma_weights)

    policy = StateStdGaussianPolicy(approximator, sigma)

    # Agent
    optimizer = AdaptiveOptimizer(eps=1e-2)
    agent = alg(mdp.info, policy, optimizer=optimizer)

    # Train
    core = Core(agent, mdp, logger=logger)

    dataset = core.evaluate(n_episodes=ep_per_run)
    J = dataset.discounted_return.mean()

    logger.log_evaluation(0, J=J)

    for i in trange(n_epochs, leave=False):
        core.learn(n_episodes=n_iterations * ep_per_run,
                   n_episodes_per_fit=ep_per_run)
        dataset = core.evaluate(n_episodes=ep_per_run)
        J = dataset.discounted_return.mean()

        logger.log_evaluation(i + 1, J=J)


if __name__ == '__main__':
    algs = [REINFORCE, GPOMDP, eNAC]

    for alg in algs:
        experiment(alg, n_epochs=10, n_iterations=4, ep_per_run=25)
