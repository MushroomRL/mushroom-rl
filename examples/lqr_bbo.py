import numpy as np
from tqdm import tqdm, trange

from mushroom_rl.algorithms.policy_search import RWR, PGPE, REPS, ConstrainedREPS, MORE
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.core import Core, Logger
from mushroom_rl.distributions import GaussianCholeskyDistribution
from mushroom_rl.environments import LQR
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.rl_utils.optimizers import AdaptiveOptimizer


"""
This script aims to replicate the experiments on the LQR MDP using episode-based
policy search algorithms, also known as Black Box policy search algorithms.

"""

tqdm.monitor_interval = 0


def experiment(alg, params, n_epochs, fit_per_epoch, ep_per_fit):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

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
    agent = alg(mdp.info, distribution, policy, **params)

    # Train
    core = Core(agent, mdp)
    dataset_eval = core.evaluate(n_episodes=ep_per_fit)
    J = np.mean(dataset_eval.discounted_return)
    logger.epoch_info(0, J=J, distribution_parameters=distribution.get_parameters())

    for i in trange(n_epochs, leave=False):
        core.learn(n_episodes=fit_per_epoch * ep_per_fit,
                   n_episodes_per_fit=ep_per_fit)
        dataset_eval = core.evaluate(n_episodes=ep_per_fit)
        J = np.mean(dataset_eval.discounted_return)
        logger.epoch_info(i+1, J=J, distribution_parameters=distribution.get_parameters())


if __name__ == '__main__':
    optimizer = AdaptiveOptimizer(eps=0.05)

    algs = [REPS, RWR, PGPE, ConstrainedREPS, MORE]
    params = [{'eps': 0.5}, {'beta': 0.7}, {'optimizer': optimizer}, {'eps':0.5, 'kappa':5}, {'eps': 0.5}]

    for alg, params in zip(algs, params):
        experiment(alg, params, n_epochs=4, fit_per_epoch=10, ep_per_fit=100)
