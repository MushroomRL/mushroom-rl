"""
This script shows how to run the MushroomRL Air Hockey environment, solving the hitting task with SAC.

"""
import argparse

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import trange

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.mujoco_envs.air_hockey import AirHockeyHit
from mushroom_rl.approximators.parametric.networks import ActorNetwork, CriticNetwork
from mushroom_rl.utils.torch_utils import TorchUtils


def experiment(n_epochs, n_steps, n_steps_test, render=True, use_cuda=False, seed=None):
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    if use_cuda:
        assert torch.cuda.is_available(), 'CUDA was requested, but it is not available on this machine.'

    TorchUtils.set_default_device('cuda:0' if use_cuda else 'cpu')

    # MDP
    horizon = 120
    gamma = 0.99
    mdp = AirHockeyHit(n_intermediate_steps=4, gamma=gamma, horizon=horizon)

    logger = Logger(SAC.name(), results_dir=None)
    logger.log_experiment_info(SAC, mdp, n_epochs=n_epochs, n_steps=n_steps, n_steps_test=n_steps_test)

    # Settings
    initial_replay_size = 5000
    max_replay_size = 200000
    batch_size = 64
    n_features = 128
    warmup_transitions = 10000
    tau = 0.001
    lr_alpha = 3e-4

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = dict(network=ActorNetwork,
                           n_features=n_features,
                           input_shape=actor_input_shape,
                           output_shape=mdp.info.action_space.shape)
    actor_sigma_params = dict(network=ActorNetwork,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=mdp.info.action_space.shape)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 1e-4}}

    critic_input_shape = [actor_input_shape, mdp.info.action_space.shape]
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 1e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,))

    # Agent
    agent = SAC(mdp.info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params,
                batch_size, initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha)

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)

    J = dataset.discounted_return.mean()
    R = dataset.undiscounted_return.mean()
    E = agent.policy.entropy(torch.from_numpy(dataset.state)).item()

    logger.log_evaluation(0, J=J, R=R, entropy=E)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)

        J = dataset.discounted_return.mean()
        R = dataset.undiscounted_return.mean()
        E = agent.policy.entropy(torch.from_numpy(dataset.state)).item()

        logger.log_evaluation(n + 1, J=J, R=R, entropy=E)

    if render:
        logger.info('Press a button to visualize the air hockey table')
        input()
        core.evaluate(n_episodes=5, render=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-render', action='store_false', dest='render', help='skip the final visualization')
    parser.add_argument('--use-cuda', action='store_true',  help='run on the GPU instead of the CPU')
    parser.add_argument('--seed', type=int, default=None, help='seed of the experiment, random when not given')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    experiment(n_epochs=100, n_steps=4000, n_steps_test=3000, render=args.render, use_cuda=args.use_cuda,
               seed=args.seed)
