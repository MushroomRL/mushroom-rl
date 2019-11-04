import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom.algorithms.actor_critic import SAC
from mushroom.core import Core
from mushroom.environments.gym_env import Gym


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h = nn.Linear(n_input, n_output)

        nn.init.xavier_uniform_(self._h.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        q = F.relu(self._h(state_action))

        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h = nn.Linear(n_input, n_output)

        nn.init.xavier_uniform_(self._h.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, state):
        return F.relu(self._h(torch.squeeze(state, 1).float()))


def test_sac():
    # MDP
    horizon = 200
    gamma = 0.99
    mdp = Gym('Pendulum-v0', horizon, gamma)
    mdp.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # Settings
    initial_replay_size = 64
    max_replay_size = 50000
    batch_size = 64
    n_features = 64
    warmup_transitions = 100
    tau = 0.005
    lr_alpha = 3e-4

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = dict(network=ActorNetwork,
                           n_features=n_features,
                           input_shape=actor_input_shape,
                           output_shape=mdp.info.action_space.shape,
                           use_cuda=False)
    actor_sigma_params = dict(network=ActorNetwork,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=mdp.info.action_space.shape,
                              use_cuda=False)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 3e-4}}

    critic_input_shape = (
    actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         use_cuda=False)

    # Agent
    agent = SAC(mdp.info, batch_size, initial_replay_size, max_replay_size,
                warmup_transitions, tau, lr_alpha, actor_mu_params,
                actor_sigma_params, actor_optimizer, critic_params,
                critic_fit_params=None)

    # Algorithm
    core = Core(agent, mdp)

    core.learn(n_steps=initial_replay_size,
               n_steps_per_fit=initial_replay_size)

    w = agent.policy.get_weights()
    w_test = np.array([1.6995193, -0.73282796, 1.2989079, -0.26830125,
                       0.5094043, -0.5001421, -0.18989229, -0.30646914])

    assert np.allclose(w, w_test)
