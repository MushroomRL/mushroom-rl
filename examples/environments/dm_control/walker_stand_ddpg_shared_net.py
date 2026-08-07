"""
Simple script to run the dm_control walker stand-up task from pixels with DDPG.

The actor and the critic share the convolution layers embedding the pixel observation, so the features of
the image are learned once instead of twice.

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


class StateEmbedding(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        history_length, n_channels = input_shape[0], input_shape[1]
        n_input = history_length * n_channels
        self._obs_shape = (n_input,) + input_shape[2:]

        self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=3)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_out_size = TorchUtils.compute_flat_output_size(nn.Sequential(self._h1, self._h2, self._h3),
                                                            self._obs_shape)
        self._output_shape = (conv_out_size,)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, state):
        h = state.view(-1, *self._obs_shape).float() / 255.
        h = F.relu(self._h1(h))
        h = F.relu(self._h2(h))
        h = F.relu(self._h3(h))
        return h

    @property
    def output_shape(self):
        return self._output_shape


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        assert isinstance(input_shape, list) and len(input_shape) == 2, \
            'CriticNetwork requires input_shape=[state_shape, action_shape].'

        self._state_embedding = kwargs['embedding']

        n_input = input_shape[0][-1] + input_shape[1][-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        h = self._state_embedding(state)
        h = h.view(-1, *self._state_embedding.output_shape)
        h = torch.cat((h, action.float()), dim=1)
        h = F.relu(self._h1(h))
        q = self._h2(h)
        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        self._state_embedding = kwargs['embedding']

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        h = self._state_embedding(state)
        h = h.view(-1, *self._state_embedding.output_shape)
        h = F.relu(self._h1(h))
        a = self._h2(h)
        return a.squeeze()


def experiment(n_epochs, n_steps, n_steps_test, render=True, use_cuda=False, seed=None):
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    if use_cuda:
        assert torch.cuda.is_available(), 'CUDA was requested, but it is not available on this machine.'

    TorchUtils.set_default_device('cuda:0' if use_cuda else 'cpu')

    # MDP
    horizon = 500
    gamma = 0.99
    mdp = DMControl('walker', 'stand', horizon, gamma, use_pixels=True, pixels_width=84, pixels_height=84)

    logger = Logger(DDPG.name(), results_dir=None)
    logger.log_experiment_info(DDPG, mdp, n_epochs=n_epochs, n_steps=n_steps, n_steps_test=n_steps_test)

    # Policy
    policy_class = OrnsteinUhlenbeckPolicy
    policy_params = dict(sigma=torch.ones(1) * .2, theta=.15, dt=1e-2)

    # Settings
    initial_replay_size = 500
    max_replay_size = 5000
    batch_size = 256
    n_features = 80
    tau = .001
    history_length = 3

    # Approximator
    embedding = StateEmbedding((history_length,) + mdp.info.observation_space.shape)

    actor_input_shape = embedding.output_shape
    actor_params = dict(network=ActorNetwork,
                        n_features=n_features,
                        input_shape=actor_input_shape,
                        output_shape=mdp.info.action_space.shape,
                        embedding=embedding)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 1e-4}}

    critic_input_shape = [actor_input_shape, mdp.info.action_space.shape]
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 1e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         embedding=embedding)

    # Agent
    agent = DDPG(mdp.info, policy_class, policy_params,
                 actor_params, actor_optimizer, critic_params,
                 batch_size, initial_replay_size, max_replay_size,
                 tau, history_length=history_length)

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    # Fill the replay memory with random samples
    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    R = dataset.undiscounted_return.mean()

    logger.log_evaluation(0, R=R)

    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        R = dataset.undiscounted_return.mean()

        logger.log_evaluation(n + 1, R=R)

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

    experiment(n_epochs=40, n_steps=1000, n_steps_test=2000, render=args.render, use_cuda=args.use_cuda, seed=args.seed)
