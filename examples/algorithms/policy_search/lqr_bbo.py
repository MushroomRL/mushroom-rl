"""
This script shows how to solve the LQR problem with episode-based policy search algorithms, also known as
black box policy search algorithms.

"""
import numpy as np

from tqdm import trange

from mushroom_rl.algorithms.policy_search import RWR, PGPE, REPS, ConstrainedREPS, MORE
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.core import Core, Logger
from mushroom_rl.distributions import GaussianCholeskyDistribution
from mushroom_rl.environments import LQR
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.rl_utils.optimizers import AdaptiveOptimizer


def experiment(alg, params, n_epochs, fit_per_epoch, ep_per_fit, seed=None):
    np.random.seed(seed)

    # MDP
    mdp = LQR.generate(dimensions=1)

    logger = Logger(alg.name(), results_dir=None)
    logger.log_experiment_info(alg, mdp, n_epochs=n_epochs, fit_per_epoch=fit_per_epoch,
                               ep_per_fit=ep_per_fit, **params)

    # Policy
    approximator = LinearApproximator(input_shape=mdp.info.observation_space.shape,
                                      output_shape=mdp.info.action_space.shape)

    policy = DeterministicPolicy(mu=approximator)

    mu = np.zeros(policy.weights_size)
    sigma = 1e-3 * np.eye(policy.weights_size)
    distribution = GaussianCholeskyDistribution(mu, sigma)

    # Agent
    agent = alg(mdp.info, distribution, policy, **params)

    # Train
    core = Core(agent, mdp, logger=logger)

    dataset = core.evaluate(n_episodes=ep_per_fit)
    J = dataset.discounted_return.mean()

    logger.log_evaluation(0, J=J, distribution_parameters=distribution.get_parameters())

    for i in trange(n_epochs, leave=False):
        core.learn(n_episodes=fit_per_epoch * ep_per_fit,
                   n_episodes_per_fit=ep_per_fit)
        dataset = core.evaluate(n_episodes=ep_per_fit, greedy=True)
        J = dataset.discounted_return.mean()

        logger.log_evaluation(i + 1, J=J, distribution_parameters=distribution.get_parameters())


if __name__ == '__main__':
    algs_params = [
        (REPS, {'eps': 0.5}),
        (RWR, {'beta': 0.3}),
        (PGPE, {'optimizer': AdaptiveOptimizer(eps=0.05)}),
        (ConstrainedREPS, {'eps': 0.5, 'kappa': 5}),
        (MORE, {'eps': 0.5}),
        ]

    for alg, params in algs_params:
        experiment(alg, params, n_epochs=4, fit_per_epoch=10, ep_per_fit=100)
