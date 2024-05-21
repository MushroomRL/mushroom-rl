import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import DDPG
from mushroom_rl.core import Core
from mushroom_rl.environments.dm_control_env import DMControl
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy

"""
Simple script to run DMControl walker stand-up task from pixels with DDPG.

"""


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input_obs = input_shape[0]
        n_input_act = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Conv2d(n_input_obs, 32, kernel_size=8, stride=3)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        dummy_obs = torch.zeros(1, *input_shape[:-1])
        conv_out_size = np.prod(self._h3(self._h2(self._h1(dummy_obs))).shape)
        self._h4 = nn.Linear(conv_out_size + n_input_act, n_features)
        self._h5 = nn.Linear(n_features, 1)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h5.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        h = F.relu(self._h1(state.squeeze().float() / 255.))
        h = F.relu(self._h2(h))
        h = F.relu(self._h3(h))
        h = torch.cat((h.view(1, state.shape[1], -1), action.float()), dim=2)
        h = F.relu(self._h4(h.view(1, state.shape[1], -1)))
        q = self._h5(h)

        return torch.squeeze(q, 2)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[0]
        n_output = output_shape[0]

        self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=3)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        dummy_obs = torch.zeros(1, *input_shape)
        conv_out_size = np.prod(self._h3(self._h2(self._h1(dummy_obs))).shape)
        self._h4 = nn.Linear(conv_out_size, n_features)
        self._h5 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h5.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        h = F.relu(self._h1(state.float() / 255.))
        h = F.relu(self._h2(h))
        h = F.relu(self._h3(h))
        h = F.relu(self._h4(h.view(state.shape[0], -1)))
        a = self._h5(h)

        return a


def experiment():
    # MDP
    horizon = 500
    gamma = 0.99
    mdp = DMControl('walker', 'stand', horizon, gamma, use_pixels=True)

    # Policy
    policy_class = OrnsteinUhlenbeckPolicy
    policy_params = dict(sigma=np.ones(1) * .2, theta=.15, dt=1e-2)

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
                       'params': {'lr': 1e-5}}

    critic_input_shape = actor_input_shape + mdp.info.action_space.shape
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 1e-3}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,))

    # Agent
    agent = DDPG(mdp.info, policy_class, policy_params,
                 actor_params, actor_optimizer, critic_params,
                 batch_size, initial_replay_size, max_replay_size,
                 tau)

    # Algorithm
    core = Core(agent, mdp)

    # Fill the replay memory with random samples
    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    # RUN
    n_epochs = 40
    n_steps = 1000
    n_steps_test = 2000

    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    J = dataset.undiscounted_return
    print('Epoch: 0')
    print('J: ', np.mean(J))

    for n in range(n_epochs):
        print('Epoch: ', n+1)
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        J = dataset.undiscounted_return
        print('J: ', np.mean(J))


if __name__ == '__main__':
    experiment()
