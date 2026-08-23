"""
This script shows how to run the MushroomRL MuJoCo locomotion environments, solving them with PPO.

"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange

from mushroom_rl.core import Core, Logger
from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.environments import Ant, HalfCheetah, Hopper, Walker2D
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.approximators.parametric.networks import FeedForwardNetwork, ActorNetwork
from mushroom_rl.utils.torch_utils import TorchUtils
from mushroom_rl.utils.experiments import select_class


def get_environments():
    return [Ant, HalfCheetah, Hopper, Walker2D]


def experiment(env, n_epochs, n_steps, n_steps_per_fit, n_episodes_test, render=True, use_cuda=False, seed=None):
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    if use_cuda:
        assert torch.cuda.is_available(), 'CUDA was requested, but it is not available on this machine.'

    TorchUtils.set_default_device('cuda:0' if use_cuda else 'cpu')

    # MDP
    mdp = env()

    logger = Logger(f'{PPO.name()}_{mdp.name()}', results_dir=None)
    logger.log_experiment_info(PPO, mdp, n_epochs=n_epochs, n_steps=n_steps, n_steps_per_fit=n_steps_per_fit,
                               n_episodes_test=n_episodes_test)

    # Settings
    actor_lr = 3e-4
    critic_lr = 3e-4
    n_features = 32
    batch_size = 64
    n_epochs_policy = 10
    eps = 0.2
    lam = 0.95
    std_0 = 1.0

    # Policy
    policy = GaussianTorchPolicy(ActorNetwork,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 std_0=std_0, n_features=n_features, gain_scale=0.1)

    # Agent
    critic_params = dict(network=FeedForwardNetwork,
                         optimizer={'class': optim.Adam, 'params': {'lr': critic_lr}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         gain_scale=0.1,
                         batch_size=batch_size,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    agent = PPO(mdp.info, policy, critic_params=critic_params,
                actor_optimizer={'class': optim.Adam, 'params': {'lr': actor_lr}},
                n_epochs_policy=n_epochs_policy,
                batch_size=batch_size,
                eps_ppo=eps,
                lam=lam)

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    # RUN
    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

    J = dataset.discounted_return.mean()
    R = dataset.undiscounted_return.mean()
    E = agent.policy.entropy().item()

    logger.log_evaluation(0, J=J, R=R, entropy=E)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

        J = dataset.discounted_return.mean()
        R = dataset.undiscounted_return.mean()
        E = agent.policy.entropy().item()

        logger.log_evaluation(it + 1, J=J, R=R, entropy=E)

    if render:
        logger.info('Press a button to visualize the robot')
        input()
        core.evaluate(n_episodes=5, render=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', choices=[env.name() for env in get_environments()], default=Ant.name(),
                        help='the locomotion environment to solve')
    parser.add_argument('--no-render', action='store_false', dest='render', help='skip the final visualization')
    parser.add_argument('--use-cuda', action='store_true', help='run on the GPU instead of the CPU')
    parser.add_argument('--seed', type=int, default=None, help='seed of the experiment, random when not given')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    env = select_class(args.env, get_environments())

    experiment(env=env, n_epochs=50, n_steps=30000, n_steps_per_fit=2000, n_episodes_test=10,
               render=args.render, use_cuda=args.use_cuda, seed=args.seed)
