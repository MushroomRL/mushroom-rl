"""
This script shows how to use the observation normalization preprocessor and the dataset monitoring callback.

"""
import numpy as np

from tqdm import trange

from mushroom_rl.algorithms.policy_search import REINFORCE
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import LQR
from mushroom_rl.policy import StateStdGaussianPolicy
from mushroom_rl.rl_utils.optimizers import AdaptiveOptimizer
from mushroom_rl.rl_utils.preprocessors import MinMaxPreprocessor
from mushroom_rl.utils.callbacks import DatasetMonitor


def experiment(n_epochs, n_iterations, ep_per_run, seed=None):
    np.random.seed(seed)

    # MDP
    mdp = LQR.generate(dimensions=2, max_pos=10., max_action=5., episodic=True)

    logger = Logger('monitoring_and_normalization', results_dir=None)
    logger.log_experiment_info(REINFORCE, mdp, n_epochs=n_epochs, n_iterations=n_iterations,
                               ep_per_run=ep_per_run)

    # Policy
    approximator = LinearApproximator(input_shape=mdp.info.observation_space.shape,
                                      output_shape=mdp.info.action_space.shape)

    sigma = LinearApproximator(input_shape=mdp.info.observation_space.shape,
                               output_shape=mdp.info.action_space.shape)

    sigma_weights = 2 * np.ones(sigma.weights_size)
    sigma.set_weights(sigma_weights)

    policy = StateStdGaussianPolicy(approximator, sigma)

    # Agent
    agent = REINFORCE(mdp.info, policy, optimizer=AdaptiveOptimizer(eps=.01))

    # normalization preprocessor
    agent.add_core_preprocessor(MinMaxPreprocessor(mdp_info=mdp.info))

    # monitoring callback
    monitor = DatasetMonitor(mdp.info, obs_normalized=True)

    # Train
    core = Core(agent, mdp, logger=logger, callback_step=monitor)

    for n in trange(n_epochs, leave=False):
        core.learn(n_episodes=n_iterations * ep_per_run,
                   n_episodes_per_fit=ep_per_run)
        dataset = core.evaluate(n_episodes=ep_per_run, render=False)
        J = dataset.discounted_return.mean()

        logger.log_evaluation(n + 1, J=J)


if __name__ == '__main__':
    experiment(n_epochs=10, n_iterations=10, ep_per_run=100)
