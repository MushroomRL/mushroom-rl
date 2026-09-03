"""
Simple script to solve the Pendulum problem with SAC.

It also shows how to save the best agent found during the run, and how to load it back to skip the training.

"""
import argparse

from pathlib import Path

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import trange

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gymnasium
from mushroom_rl.utils import TorchUtils
from mushroom_rl.approximators.parametric.networks import ActorNetwork, CriticNetwork
from mushroom_rl.utils.experiments import get_log_dir


def build_agent(mdp, batch_size, initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha,
                n_features):
    """
    Build the SAC agent, with a Gaussian actor and a pair of critics.

    """
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
                       'params': {'lr': 3e-4}}

    critic_input_shape = [actor_input_shape, mdp.info.action_space.shape]
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,))

    return SAC(mdp.info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params,
               batch_size, initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha)


def experiment(n_epochs, n_steps, n_steps_test, save=False, agent_dir=None, render=True, use_cuda=False, seed=None):
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

    logger = Logger(SAC.name(), results_dir=get_log_dir(__file__) if save else None)
    logger.log_experiment_info(SAC, mdp, n_epochs=n_epochs, n_steps=n_steps, n_steps_test=n_steps_test)

    # Settings
    initial_replay_size = 64
    max_replay_size = 50000
    batch_size = 64
    n_features = 64
    warmup_transitions = 100
    tau = 0.005
    lr_alpha = 3e-4

    # Agent
    if agent_dir is not None:
        agent = SAC.load(agent_dir / 'agent-best.msh')
    else:
        agent = build_agent(mdp, batch_size, initial_replay_size, max_replay_size, warmup_transitions,
                            tau, lr_alpha, n_features)

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)

    J = dataset.discounted_return.mean()
    R = dataset.undiscounted_return.mean()
    E = agent.policy.entropy(agent.history_manager.parse_state(dataset)).item()

    logger.log_evaluation(0, J=J, R=R, entropy=E)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)

        J = dataset.discounted_return.mean()
        R = dataset.undiscounted_return.mean()
        E = agent.policy.entropy(agent.history_manager.parse_state(dataset)).item()

        logger.log_evaluation(n + 1, J=J, R=R, entropy=E)

        if save:
            logger.log_best_agent(agent, J)

    if render:
        logger.info('Press a button to visualize the pendulum')
        input()
        core.evaluate(n_episodes=5, render=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save', action='store_true',
                        help='save the best agent of the run, as agent-best.msh in the log directory')
    parser.add_argument('--load', type=Path, default=None, metavar='DIR',
                        help='log directory of a previous run, whose best agent is the one to start from')
    parser.add_argument('--no-render', action='store_false', dest='render', help='skip the final visualization')
    parser.add_argument('--use-cuda', action='store_true',  help='run on the GPU instead of the CPU')
    parser.add_argument('--seed', type=int, default=None, help='seed of the experiment, random when not given')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    experiment(n_epochs=40, n_steps=1000, n_steps_test=2000, save=args.save, agent_dir=args.load,
               render=args.render, use_cuda=args.use_cuda, seed=args.seed)
