"""
Simple script to solve the Acrobot problem with DQN.

"""
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import trange

from mushroom_rl.algorithms.value import DQN
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gymnasium
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter, LinearParameter
from mushroom_rl.approximators.parametric.networks import QNetwork
from mushroom_rl.utils.torch_utils import TorchUtils


def experiment(n_epochs, n_steps, n_steps_test, render=True, use_cuda=False, seed=None):
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    if use_cuda:
        assert torch.cuda.is_available(), 'CUDA was requested, but it is not available on this machine.'

    TorchUtils.set_default_device('cuda:0' if use_cuda else 'cpu')

    # MDP
    horizon = 1000
    gamma = 0.99
    mdp = Gymnasium('Acrobot-v1', horizon, gamma, headless=False)
    mdp.seed(seed)

    logger = Logger(DQN.name(), results_dir=None)
    logger.log_experiment_info(DQN, mdp, n_epochs=n_epochs, n_steps=n_steps, n_steps_test=n_steps_test)

    # Policy
    epsilon = LinearParameter(value=1., threshold_value=.01, n=5000)
    epsilon_random = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon_random, backend='torch')

    # Settings
    initial_replay_size = 500
    max_replay_size = 5000
    target_update_frequency = 100
    batch_size = 200
    n_features = 80
    train_frequency = 1

    # Approximator
    input_shape = mdp.info.observation_space.shape
    approximator_params = dict(network=QNetwork,
                               optimizer={'class': optim.Adam,
                                          'params': {'lr': .001}},
                               loss=F.smooth_l1_loss,
                               n_features=n_features,
                               input_shape=input_shape,
                               output_shape=mdp.info.action_space.size,
                               n_actions=mdp.info.action_space.n)

    # Agent
    agent = DQN(mdp.info, pi, approximator_params=approximator_params, batch_size=batch_size,
                initial_replay_size=initial_replay_size,
                max_replay_size=max_replay_size,
                target_update_frequency=target_update_frequency)

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False, greedy=True)
    R = dataset.undiscounted_return.mean()

    logger.log_evaluation(0, R=R)

    pi.set_epsilon(epsilon)
    for n in trange(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=train_frequency)
        dataset = core.evaluate(n_steps=n_steps_test, render=False, greedy=True)
        R = dataset.undiscounted_return.mean()

        logger.log_evaluation(n + 1, R=R)

    if render:
        logger.info('Press a button to visualize acrobot')
        input()
        core.evaluate(n_episodes=5, render=True, greedy=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-render', action='store_true', help='skip the final visualization')
    parser.add_argument('--use-cuda', action='store_true', help='run on the GPU instead of the CPU')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    experiment(n_epochs=20, n_steps=1000, n_steps_test=2000, render=not args.no_render, use_cuda=args.use_cuda)
