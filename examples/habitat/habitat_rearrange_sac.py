import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.habitat_env import *

from tqdm import trange


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


def experiment(alg, n_epochs, n_steps, n_episodes_test):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    # MDP
    gamma = 0.99
    habitat_root_path = Habitat.root_path()
    config_file = os.path.join(habitat_root_path,
        'habitat_baselines/config/rearrange/rl_pick.yaml')
    base_config_file = os.path.join(habitat_root_path,
        'configs/tasks/rearrange/pick.yaml')
    wrapper = 'HabitatRearrangeWrapper'
    mdp = Habitat(wrapper, config_file, base_config_file, gamma=gamma)

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
                           output_shape=mdp.info.action_space.shape)
    actor_sigma_params = dict(network=ActorNetwork,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=mdp.info.action_space.shape)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 3e-4}}


    critic_input_shape = actor_input_shape + mdp.info.action_space.shape
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,))

    # Agent
    agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                actor_optimizer, critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None)

    # Algorithm
    core = Core(agent, mdp)

    # RUN
    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

    J = np.mean(dataset.discounted_return)
    R = np.mean(dataset.undiscounted_return)
    E = agent.policy.entropy(dataset.state).item()

    logger.epoch_info(0, J=J, R=R, entropy=E)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

        J = np.mean(dataset.discounted_return)
        R = np.mean(dataset.undiscounted_return)
        E = agent.policy.entropy(dataset.state).item()

        logger.epoch_info(n+1, J=J, R=R, entropy=E)

    logger.info('Press a button to visualize the robot')
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':
    algs = [
        SAC
    ]

    for alg in algs:
        experiment(alg=alg, n_epochs=50, n_steps=1000, n_episodes_test=5)
