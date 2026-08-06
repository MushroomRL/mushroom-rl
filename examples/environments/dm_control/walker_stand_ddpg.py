"""
Simple script to run the dm_control walker stand-up task from pixels with DDPG.

"""
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import trange

from mushroom_rl.algorithms.actor_critic import DDPG
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import DMControl
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy
from mushroom_rl.utils.torch_utils import TorchUtils


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        assert isinstance(input_shape, list) and len(input_shape) == 2, \
            'CriticNetwork requires input_shape=[state_shape, action_shape].'

        n_input_obs = input_shape[0][0]
        n_input_act = input_shape[1][0]
        n_output = output_shape[0]

        self._h1 = nn.Conv2d(n_input_obs, 32, kernel_size=8, stride=3)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        conv_out_size = TorchUtils.compute_flat_output_size(nn.Sequential(self._h1, self._h2, self._h3), input_shape[0])
        self._h4 = nn.Linear(conv_out_size + n_input_act, n_features)
        self._h5 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h5.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        h = F.relu(self._h1(state.float() / 255.))
        h = F.relu(self._h2(h))
        h = F.relu(self._h3(h))
        h = torch.cat((h.view(state.shape[0], -1), action.float()), dim=1)
        h = F.relu(self._h4(h))
        q = self._h5(h)

        return torch.squeeze(q, -1)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[0]
        n_output = output_shape[0]

        self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=3)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        conv_out_size = TorchUtils.compute_flat_output_size(nn.Sequential(self._h1, self._h2, self._h3), input_shape)
        self._h4 = nn.Linear(conv_out_size, n_features)
        self._h5 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h5.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        h = F.relu(self._h1(state.float() / 255.))
        h = F.relu(self._h2(h))
        h = F.relu(self._h3(h))
        h = F.relu(self._h4(h.view(state.shape[0], -1)))
        a = self._h5(h)

        return a


def experiment(n_epochs, n_steps, n_episode_test, render=True, use_cuda=False, seed=None):
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    if use_cuda:
        assert torch.cuda.is_available(), 'CUDA was requested, but it is not available on this machine.'

    TorchUtils.set_default_device('cuda:0' if use_cuda else 'cpu')

    # MDP
    horizon = 500
    gamma = 0.99
    mdp = DMControl('walker', 'stand', horizon, gamma, use_pixels=True)

    logger = Logger(DDPG.name(), results_dir=None)
    logger.log_experiment_info(DDPG, mdp, n_epochs=n_epochs, n_steps=n_steps, n_episodes_test=n_episode_test)

    # Policy
    policy_class = OrnsteinUhlenbeckPolicy
    policy_params = dict(sigma=torch.ones(1) * .2, theta=.15, dt=1e-2)

    # Settings
    initial_replay_size = 5000
    max_replay_size = 100000
    batch_size = 200
    n_features = 256
    tau = .001

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_params = dict(network=ActorNetwork,
                        n_features=n_features,
                        input_shape=actor_input_shape,
                        output_shape=mdp.info.action_space.shape)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 1e-6}}

    critic_input_shape = [actor_input_shape, mdp.info.action_space.shape]
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 1e-6}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,))

    # Agent
    agent = DDPG(mdp.info, policy_class, policy_params, actor_params, actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, tau)

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    # Fill the replay memory with random samples
    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    # RUN
    dataset = core.evaluate(n_episodes=n_episode_test, render=False)
    J = dataset.discounted_return.mean()
    R = dataset.undiscounted_return.mean()

    logger.log_evaluation(0, J=J, R=R)

    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_episodes=n_episode_test, render=False)
        J = dataset.discounted_return.mean()
        R = dataset.undiscounted_return.mean()

        logger.log_evaluation(n + 1, J=J, R=R)

    if render:
        logger.info('Press a button to visualize the learning results')
        input()
        core.evaluate(n_episodes=5, render=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-render', action='store_false', dest='render', help='skip the final visualization')
    parser.add_argument('--use-cuda', action='store_true', help='Flag specifying whether to use the GPU.')

    parser.add_argument('--seed', type=int, default=None,
                        help='seed of the experiment, random when not given')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    experiment(n_epochs=40, n_steps=10000, n_episode_test=5, render=args.render, use_cuda=args.use_cuda, seed=args.seed)
