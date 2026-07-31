"""
Simple script to solve the Pendulum problem with DDPG and TD3.

"""
import argparse

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import trange

from mushroom_rl.algorithms.actor_critic import DDPG, TD3
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gymnasium
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy
from mushroom_rl.utils import select_class
from mushroom_rl.approximators.parametric.networks import ActorNetwork, CriticNetwork
from mushroom_rl.utils.torch_utils import TorchUtils


def get_algorithms():
    return [DDPG, TD3]


def experiment(alg, n_epochs, n_steps, n_steps_test, render=True, use_cuda=False, seed=None):
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    if use_cuda:
        assert torch.cuda.is_available(), 'CUDA was requested, but it is not available on this machine.'

    TorchUtils.set_default_device('cuda:0' if use_cuda else 'cpu')

    # MDP
    horizon = 200
    gamma = 0.99
    mdp = Gymnasium('Pendulum-v1', horizon, gamma, headless=False)
    mdp.seed(seed)

    logger = Logger(alg.name(), results_dir=None)
    logger.log_experiment_info(alg, mdp, n_epochs=n_epochs, n_steps=n_steps, n_steps_test=n_steps_test)

    # Policy
    policy_class = OrnsteinUhlenbeckPolicy
    policy_params = dict(sigma=torch.ones(1) * .2, theta=.15, dt=1e-2)

    # Settings
    initial_replay_size = 500
    max_replay_size = 5000
    batch_size = 200
    n_features = 80
    tau = .001

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_params = dict(network=ActorNetwork,
                        n_features=n_features,
                        input_shape=actor_input_shape,
                        output_shape=mdp.info.action_space.shape)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': .001}}

    critic_input_shape = [actor_input_shape, mdp.info.action_space.shape]
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': .001}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,))

    # Agent
    agent = alg(mdp.info, policy_class, policy_params,
                actor_params, actor_optimizer, critic_params, batch_size,
                initial_replay_size, max_replay_size, tau)

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)

    J = dataset.discounted_return.mean()
    R = dataset.undiscounted_return.mean()

    logger.log_evaluation(0, J=J, R=R)

    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)

        J = dataset.discounted_return.mean()
        R = dataset.undiscounted_return.mean()

        logger.log_evaluation(n + 1, J=J, R=R)

    if render:
        logger.info('Press a button to visualize the pendulum')
        input()
        core.evaluate(n_episodes=5, render=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alg', choices=[alg.name() for alg in get_algorithms()], default=DDPG.name(),
                        help='the algorithm to run')
    parser.add_argument('--no-render', action='store_false', dest='render', help='skip the final visualization')

    parser.add_argument('--use-cuda', action='store_true', help='run on the GPU instead of the CPU')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    alg = select_class(args.alg, get_algorithms())

    experiment(alg=alg, n_epochs=40, n_steps=1000, n_steps_test=2000, render=args.render, use_cuda=args.use_cuda)
