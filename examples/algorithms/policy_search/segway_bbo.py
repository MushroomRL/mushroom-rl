"""
This script shows how to solve the Segway problem with episode-based policy search algorithms and a linear
deterministic policy.

"""
import numpy as np

from tqdm import trange

from mushroom_rl.algorithms.policy_search import REPS, RWR, PGPE
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.core import Core, Logger
from mushroom_rl.distributions import GaussianDiagonalDistribution
from mushroom_rl.environments import Segway
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.rl_utils.optimizers import AdaptiveOptimizer
from mushroom_rl.utils.callbacks import CollectDataset


def experiment(alg, params, n_epochs, n_episodes, n_ep_per_fit, seed=None):
    np.random.seed(seed)

    # MDP
    mdp = Segway()

    logger = Logger(alg.name(), results_dir=None)
    logger.log_experiment_info(alg, mdp, n_epochs=n_epochs, n_episodes=n_episodes,
                               n_ep_per_fit=n_ep_per_fit, **params)

    # Policy
    approximator = LinearApproximator(input_shape=mdp.info.observation_space.shape,
                                      output_shape=mdp.info.action_space.shape)

    n_weights = approximator.weights_size
    mu = np.zeros(n_weights)
    sigma = 2e-0 * np.ones(n_weights)
    policy = DeterministicPolicy(approximator)
    dist = GaussianDiagonalDistribution(mu, sigma)

    # Agent
    agent = alg(mdp.info, dist, policy, **params)

    # Train
    dataset_callback = CollectDataset()
    core = Core(agent, mdp, logger=logger, callbacks_fit=[dataset_callback])

    dataset = core.evaluate(n_episodes=n_episodes)
    J = dataset.discounted_return.mean()
    p = dist.get_parameters()

    logger.log_evaluation(0, J=J, mu=p[:n_weights], sigma=p[n_weights:])

    for i in trange(n_epochs, leave=False):
        core.learn(n_episodes=n_episodes, n_episodes_per_fit=n_ep_per_fit, render=False)
        J = dataset_callback.get().discounted_return.mean()
        dataset_callback.clean()
        p = dist.get_parameters()

        logger.log_evaluation(i + 1, J=J, mu=p[:n_weights], sigma=p[n_weights:])

    # Visualize the final policy
    core.evaluate(n_episodes=3, render=True)


if __name__ == '__main__':
    algs_params = [
        (REPS, {'eps': 0.05}),
        (RWR, {'beta': 0.01}),
        (PGPE, {'optimizer': AdaptiveOptimizer(eps=0.3)}),
        ]

    for alg, params in algs_params:
        experiment(alg, params, n_epochs=20, n_episodes=100, n_ep_per_fit=25)
