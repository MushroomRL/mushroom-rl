import numpy as np

from mushroom_rl.algorithms.policy_search import REINFORCE, GPOMDP, eNAC
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import LQR
from mushroom_rl.policy import StateStdGaussianPolicy
from mushroom_rl.rl_utils.optimizers import AdaptiveOptimizer

from tqdm import tqdm, trange


"""
This script aims to replicate the experiments on the LQR MDP using policy
gradient algorithms.

"""

tqdm.monitor_interval = 0


def experiment(alg, n_epochs, n_iterations, ep_per_run):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    # MDP
    mdp = LQR.generate(dimensions=2, max_action=1., max_pos=1.)

    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape)

    sigma = Regressor(LinearApproximator,
                      input_shape=mdp.info.observation_space.shape,
                      output_shape=mdp.info.action_space.shape)

    sigma_weights = 0.25 * np.ones(sigma.weights_size)
    sigma.set_weights(sigma_weights)

    policy = StateStdGaussianPolicy(approximator, sigma)

    # Agent
    optimizer = AdaptiveOptimizer(eps=1e-2)
    algorithm_params = dict(optimizer=optimizer)
    agent = alg(mdp.info, policy, **algorithm_params)

    # Train
    core = Core(agent, mdp)
    dataset_eval = core.evaluate(n_episodes=ep_per_run)
    J = np.mean(dataset_eval.discounted_return)
    logger.epoch_info(0, J=J, policy_weights=policy.get_weights().tolist())

    for i in trange(n_epochs, leave=False):
        core.learn(n_episodes=n_iterations * ep_per_run,
                   n_episodes_per_fit=ep_per_run)
        dataset_eval = core.evaluate(n_episodes=ep_per_run)
        J = np.mean(dataset_eval.discounted_return)
        logger.epoch_info(i+1, J=J, policy_weights=policy.get_weights().tolist())


if __name__ == '__main__':
    algs = [REINFORCE, GPOMDP, eNAC]

    for alg in algs:
        experiment(alg, n_epochs=10, n_iterations=4, ep_per_run=25)
