import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.utils.dataset import compute_J

from mushroom_rl.utils.preprocessors import NormalizationBoxedPreprocessor
from mushroom_rl.utils.callbacks.plot_dataset import PlotDataset


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a

def create_agent(mdp):
    # Settings
    initial_replay_size = 1000
    max_replay_size = 50000
    batch_size = 64
    n_features = 64
    warmup_transitions = 1500
    tau = 0.005
    lr_alpha = 3e-4

    use_cuda = torch.cuda.is_available()

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = dict(network=ActorNetwork,
                           n_features=n_features,
                           input_shape=actor_input_shape,
                           output_shape=mdp.info.action_space.shape,
                           use_cuda=use_cuda)
    actor_sigma_params = dict(network=ActorNetwork,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=mdp.info.action_space.shape,
                              use_cuda=use_cuda)

    actor_optimizer = {'class':  optim.Adam,
                       'params': {'lr': 3e-4}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class':  optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         use_cuda=use_cuda)

    # Agent
    agent = SAC(mdp.info, actor_mu_params, actor_sigma_params,
                actor_optimizer, critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None)
    return agent


def experiment(n_epochs, n_steps, n_steps_test, save_states_to_disk=False):
    np.random.seed()

    # MDP
    horizon = 200
    gamma = 0.99
    mdp = Gym('Pendulum-v0', horizon, gamma)

    # Agent
    agent = create_agent(mdp)

    # normalization callback
    normalizer = NormalizationBoxedPreprocessor(mdp_info=mdp.info)

    # plotting callback
    plotter = PlotDataset(mdp.info, obs_normalized=True)

    # Algorithm(with normalizing and plotting)
    core = Core(agent, mdp, callbacks=[plotter], preprocessors=[normalizer])

    # training loop
    for n in range(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        print('Epoch: ', n, '  J: ', np.mean(compute_J(dataset, gamma)))

    if save_states_to_disk:
        # save normalization / plot states to disk path
        os.makedirs("./temp/", exist_ok=True)
        normalizer.save_state("./temp/normalization_state")
        plotter.save_state("./temp/plotting_state")

        # load states from disk path
        normalizer.load_state("./temp/normalization_state")
        plotter.load_state("./temp/plotting_state")


if __name__ == '__main__':
    experiment(n_epochs=5, n_steps=1000, n_steps_test=1000,
               save_states_to_disk=False)