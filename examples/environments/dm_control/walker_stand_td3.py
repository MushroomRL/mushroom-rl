"""
Simple script to run the dm_control walker stand-up task from pixels with TD3.

The network architecture, the hyperparameters and the random shift augmentation follow the DrQ-v2 setup for
this task (Yarats et al., 2021). The encoder is not shared between actor and critic.

"""
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import trange

from mushroom_rl.algorithms.actor_critic import TD3
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import DMControl
from mushroom_rl.policy import ClippedGaussianPolicy
from mushroom_rl.rl_utils.parameters import LinearParameter
from mushroom_rl.utils.torch_utils import TorchUtils


def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight, nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)


class RandomShiftAugmentation(nn.Module):
    def __init__(self, pad):
        super().__init__()

        self._pad = pad

    def forward(self, state):
        shape = state.shape
        image = state.reshape(-1, shape[-3] * shape[-4], *shape[-2:]).float()
        n, _, h, w = image.shape
        assert h == w, 'RandomShiftAugmentation requires square images.'

        image = F.pad(image, (self._pad,) * 4, mode='replicate')

        eps = 1. / (h + 2 * self._pad)
        arange = torch.linspace(-1. + eps, 1. - eps, h + 2 * self._pad, device=image.device,
                                dtype=image.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2).unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self._pad + 1, size=(n, 1, 1, 2), device=image.device, dtype=image.dtype)
        shift *= 2. / (h + 2 * self._pad)

        shifted = F.grid_sample(image, base_grid + shift, padding_mode='zeros', align_corners=False)

        return shifted.reshape(shape)


class Encoder(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        history_length, n_channels = input_shape[0], input_shape[1]
        n_input = history_length * n_channels
        self._obs_shape = (n_input,) + input_shape[2:]

        self._h1 = nn.Conv2d(n_input, 32, kernel_size=3, stride=2)
        self._h2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self._h3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self._h4 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

        self._output_size = TorchUtils.compute_flat_output_size(
            nn.Sequential(self._h1, self._h2, self._h3, self._h4), self._obs_shape)

        self.apply(orthogonal_init)

    def forward(self, state):
        h = state.reshape(-1, *self._obs_shape).float() / 255. - .5
        h = F.relu(self._h1(h))
        h = F.relu(self._h2(h))
        h = F.relu(self._h3(h))
        h = F.relu(self._h4(h))

        return h.view(h.shape[0], -1)

    @property
    def output_size(self):
        return self._output_size


class Trunk(nn.Module):
    def __init__(self, input_size, feature_dim):
        super().__init__()

        self._h1 = nn.Linear(input_size, feature_dim)
        self._norm = nn.LayerNorm(feature_dim)

        self.apply(orthogonal_init)

    def forward(self, features):
        return torch.tanh(self._norm(self._h1(features)))


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, feature_dim, **kwargs):
        super().__init__()

        assert isinstance(input_shape, list) and len(input_shape) == 2, \
            'CriticNetwork requires input_shape=[state_shape, action_shape].'

        n_input_act = input_shape[1][0]
        n_output = output_shape[0]

        self._encoder = Encoder(input_shape[0])
        self._trunk = Trunk(self._encoder.output_size, feature_dim)
        self._h1 = nn.Linear(feature_dim + n_input_act, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        self.apply(orthogonal_init)

    def forward(self, state, action):
        h = self._trunk(self._encoder(state))
        h = torch.cat((h, action.float()), dim=1)
        h = F.relu(self._h1(h))
        h = F.relu(self._h2(h))
        q = self._h3(h)

        return torch.squeeze(q, -1)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, feature_dim, **kwargs):
        super().__init__()

        n_output = output_shape[0]

        self._encoder = Encoder(input_shape)
        self._trunk = Trunk(self._encoder.output_size, feature_dim)
        self._h1 = nn.Linear(feature_dim, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        self.apply(orthogonal_init)

    def forward(self, state):
        h = self._trunk(self._encoder(state))
        h = F.relu(self._h1(h))
        h = F.relu(self._h2(h))
        a = torch.tanh(self._h3(h))

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
    mdp = DMControl('walker', 'stand', horizon, gamma, use_pixels=True, pixels_width=84, pixels_height=84,
                    action_repeat=2)

    logger = Logger(TD3.name(), results_dir=None)
    logger.log_experiment_info(TD3, mdp, n_epochs=n_epochs, n_steps=n_steps, n_episodes_test=n_episode_test)

    # Policy
    policy_class = ClippedGaussianPolicy
    n_actions = mdp.info.action_space.shape[0]
    policy_params = dict(sigma=LinearParameter(np.ones(n_actions), threshold_value=.1 * np.ones(n_actions), n=100000),
                         low=torch.from_numpy(mdp.info.action_space.low).float(),
                         high=torch.from_numpy(mdp.info.action_space.high).float())

    # Settings
    initial_replay_size = 2000
    max_replay_size = 100000
    batch_size = 512
    n_features = 1024
    feature_dim = 50
    tau = .01
    history_length = 3

    # Approximator
    actor_input_shape = (history_length,) + mdp.info.observation_space.shape
    actor_params = dict(network=ActorNetwork,
                        n_features=n_features,
                        feature_dim=feature_dim,
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
                         feature_dim=feature_dim,
                         input_shape=critic_input_shape,
                         output_shape=(1,))

    # Agent
    agent = TD3(mdp.info, policy_class, policy_params, actor_params, actor_optimizer, critic_params, batch_size,
                initial_replay_size, max_replay_size, tau, policy_delay=1, noise_clip=.3,
                history_length=history_length)
    agent.add_fit_preprocessor(RandomShiftAugmentation(pad=4))

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    # Fill the replay memory with random samples
    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    # RUN
    dataset = core.evaluate(n_episodes=n_episode_test, render=False, greedy=True)
    J = dataset.discounted_return.mean()
    R = dataset.undiscounted_return.mean()

    logger.log_evaluation(0, J=J, R=R)

    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=2)
        dataset = core.evaluate(n_episodes=n_episode_test, render=False, greedy=True)
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

    experiment(n_epochs=40, n_steps=10000, n_episode_test=10, render=args.render, use_cuda=args.use_cuda,
               seed=args.seed)
