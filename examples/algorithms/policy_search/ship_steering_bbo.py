"""
This script shows how to solve the Ship Steering problem with episode-based policy search algorithms and
tile coding.

"""
import numpy as np

from tqdm import trange

from mushroom_rl.algorithms.policy_search import REPS, RWR, PGPE
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import ShipSteering
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.features.features import Features
from mushroom_rl.distributions import GaussianDiagonalDistribution
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.rl_utils.optimizers import AdaptiveOptimizer


def experiment(alg, params, n_epochs, fit_per_epoch, ep_per_fit, ep_test, seed=None):
    np.random.seed(seed)

    # MDP
    mdp = ShipSteering()

    logger = Logger(alg.name(), results_dir=None)
    logger.log_experiment_info(alg, mdp, n_epochs=n_epochs, fit_per_epoch=fit_per_epoch,
                               ep_per_fit=ep_per_fit, ep_test=ep_test, **params)

    # Policy
    low = np.array([0, 0, -np.pi])
    high = np.array([150, 150, np.pi])
    tilings = Tiles.generate(n_tilings=1, n_tiles=[5, 5, 6], low=low, high=high)

    phi = Features(tilings)

    approximator = LinearApproximator(input_shape=mdp.info.observation_space.shape,
                                      output_shape=mdp.info.action_space.shape, phi=phi)

    policy = DeterministicPolicy(approximator)

    mu = np.zeros(policy.weights_size)
    sigma = 4e-1 * np.ones(policy.weights_size)
    distribution = GaussianDiagonalDistribution(mu, sigma)

    # Agent
    agent = alg(mdp.info, distribution, policy, **params)

    # Train
    core = Core(agent, mdp, logger=logger)

    dataset = core.evaluate(n_episodes=ep_test)
    J = dataset.discounted_return.mean()

    logger.log_evaluation(0, J=J)

    for i in trange(n_epochs, leave=False):
        core.learn(n_episodes=fit_per_epoch * ep_per_fit,
                   n_episodes_per_fit=ep_per_fit)
        dataset = core.evaluate(n_episodes=ep_test)
        J = dataset.discounted_return.mean()

        logger.log_evaluation(i + 1, J=J)

    # Visualize the final policy
    core.evaluate(n_episodes=ep_test, render=True)


if __name__ == '__main__':
    algs_params = [
        (REPS, {'eps': 1.0}),
        (RWR, {'beta': 0.7}),
        (PGPE, {'optimizer': AdaptiveOptimizer(eps=1.5)}),
        ]

    for alg, params in algs_params:
        experiment(alg, params, n_epochs=25, fit_per_epoch=10, ep_per_fit=20, ep_test=5)
