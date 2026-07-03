import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import DDPG
from mushroom_rl.approximators.parametric.networks import ActorNetwork, CriticNetwork
from mushroom_rl.core import Core
from mushroom_rl.environments.dm_control_env import DMControl
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy


# MDP
horizon = 500
gamma = 0.99
mdp = DMControl('walker', 'stand', horizon, gamma)

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
                   'params': {'lr': 1e-5}}

critic_input_shape = [actor_input_shape, mdp.info.action_space.shape]
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
J = dataset.discounted_return
print('Epoch: 0')
print('J: ', np.mean(J))

for n in range(n_epochs):
    print('Epoch: ', n+1)
    core.learn(n_steps=n_steps, n_steps_per_fit=1)
    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    J = dataset.discounted_return
    print('J: ', np.mean(J))
