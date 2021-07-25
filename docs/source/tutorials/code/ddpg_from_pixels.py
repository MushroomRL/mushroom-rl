import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import DDPG
from mushroom_rl.core import Core
from mushroom_rl.environments.dm_control_env import DMControl
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy
from mushroom_rl.utils.dataset import compute_J


class CriticNetwork(nn.Module):
    def __init__(self, obs_shape, action_shape, n_features, **kwargs):
        super().__init__()

        n_input = obs_shape[0]
        n_output = action_shape[0]

        self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=3)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        dummy_obs = torch.zeros(1, *obs_shape)
        conv_out_size = np.prod(self._h3(self._h2(self._h1(dummy_obs))).shape)
        self._h4 = nn.Linear(conv_out_size + n_output, self.n_features)
        self._h5 = nn.Linear(self.n_features, 1)

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
        h = F.relu(self._h1(state.float() / 255.))
        h = F.relu(self._h2(h))
        h = F.relu(self._h3(h))
        h = F.relu(self._h4(h.view(state.shape[0], -1)))
        q = self._h5(h)

        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, obs_shape, action_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = obs_shape[0]
        n_output = action_shape[0]

        self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=3)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        dummy_obs = torch.zeros(1, *obs_shape)
        conv_out_size = np.prod(self._h3(self._h2(self._h1(dummy_obs))).shape)
        self._h4 = nn.Linear(conv_out_size, self.n_features)
        self._h5 = nn.Linear(self.n_features, n_output)

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
        h = torch.cat((h.view(state.shape[0], -1), action.float()), dim=1)
        h = F.relu(self._h4(h.view(state.shape[0], -1)))
        q = self._h5(h)

        return q


# MDP
horizon = 500
gamma = 0.99
gamma_eval = 1.
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
obs_shape = mdp.info.observation_space.shape
act_shape = mdp.info.action_space.shape

actor_params = dict(network=ActorNetwork,
                    n_features=n_features,
                    input_shape=obs_shape,
                    output_shape=act_shape)

actor_optimizer = {'class': optim.Adam,
                   'params': {'lr': 1e-5}}

critic_params = dict(network=CriticNetwork,
                     optimizer={'class': optim.Adam,
                                'params': {'lr': 1e-3}},
                     loss=F.mse_loss,
                     n_features=n_features,
                     obs_shape=obs_shape,
                     act_shape=act_shape)

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
J = compute_J(dataset, gamma_eval)
print('Epoch: 0')
print('J: ', np.mean(J))

for n in range(n_epochs):
    print('Epoch: ', n+1)
    core.learn(n_steps=n_steps, n_steps_per_fit=1)
    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    J = compute_J(dataset, gamma_eval)
    print('J: ', np.mean(J))
