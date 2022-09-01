import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.utils.callbacks import PlotDataset
from mushroom_rl.utils.preprocessors import MinMaxPreprocessor

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length

from mushroom_rl.environments.mujoco_envs import HumanoidGait
from mushroom_rl.environments.mujoco_envs.humanoid_gait import \
    VelocityProfile3D, RandomConstantVelocityProfile, ConstantVelocityProfile

from tqdm import trange


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(CriticNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._in = nn.Linear(n_input, n_features[0])
        self._h1 = nn.Linear(n_features[0], n_features[1])
        self._out = nn.Linear(n_features[1], n_output)

        nn.init.xavier_uniform_(self._in.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._out.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        in_feats = torch.cat((state.float(), action.float()), dim=1)
        feats = F.relu(self._in(in_feats))
        feats = F.relu(self._h1(feats))

        out = self._out(feats)
        return torch.squeeze(out)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._in = nn.Linear(n_input, n_features[0])
        self._h1 = nn.Linear(n_features[0], n_features[1])
        self._out = nn.Linear(n_features[1], n_output)

        nn.init.xavier_uniform_(self._in.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._out.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        in_feats = torch.squeeze(state, 1).float()

        feats = F.relu(self._in(in_feats))
        feats = F.relu(self._h1(feats))

        out = self._out(feats)
        return out


def create_SAC_agent(mdp, use_cuda=None):
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()

    # Settings
    actor_mu_network = ActorNetwork
    actor_sigma_network = ActorNetwork
    network_layers_actor_mu = (512, 256)
    network_layers_actor_sigma = (512, 256)
    network_layers_critic = (512, 256)

    initial_replay_size = 3000
    max_replay_size = 100000
    batch_size = 256
    warmup_transitions = 5000
    tau = 0.005

    lr_alpha = 2e-6
    lr_actor = 2e-5
    lr_critic = 4e-5
    weight_decay_actor = 0.0
    weight_decay_critic = 0.0

    target_entropy = -22.0

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = dict(network=actor_mu_network,
                           n_features=network_layers_actor_mu,
                           input_shape=actor_input_shape,
                           output_shape=mdp.info.action_space.shape,
                           use_cuda=use_cuda)

    actor_sigma_params = dict(network=actor_sigma_network,
                              n_features=network_layers_actor_sigma,
                              input_shape=actor_input_shape,
                              output_shape=mdp.info.action_space.shape,
                              use_cuda=use_cuda)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': lr_actor, 'weight_decay': weight_decay_actor}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': lr_critic, 'weight_decay': weight_decay_critic}},
                         loss=F.mse_loss,
                         n_features=network_layers_critic,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         use_cuda=use_cuda)

    # create SAC agent
    agent = SAC(mdp_info=mdp.info,
                batch_size=batch_size, initial_replay_size=initial_replay_size,
                max_replay_size=max_replay_size,
                warmup_transitions=warmup_transitions, tau=tau, lr_alpha=lr_alpha,
                actor_mu_params=actor_mu_params, actor_sigma_params=actor_sigma_params,
                actor_optimizer=actor_optimizer, critic_params=critic_params,
                target_entropy=target_entropy, critic_fit_params=None)

    return agent


def create_mdp(gamma, horizon, goal, use_muscles):
    if goal == "trajectory" or  goal == "com_vel_trajectory":
        mdp = HumanoidGait(gamma=gamma, horizon=horizon, n_intermediate_steps=10,
                           goal_reward=goal,
                           goal_reward_params=dict(use_error_terminate=True),
                           use_muscles=use_muscles,
                           obs_avg_window=1, act_avg_window=1)

    elif goal == "max_vel":
        mdp = HumanoidGait(gamma=gamma, horizon=horizon, n_intermediate_steps=10,
                           goal_reward=goal,
                           goal_reward_params=dict(traj_start=True),
                           use_muscles=use_muscles,
                           obs_avg_window=1, act_avg_window=1)

    elif goal == "vel_profile":
        velocity_profile = dict(profile_instance=VelocityProfile3D([
                RandomConstantVelocityProfile(min=0.5, max=2.0),
                ConstantVelocityProfile(0),
                ConstantVelocityProfile(0)]))

        mdp = HumanoidGait(gamma=gamma, horizon=horizon, n_intermediate_steps=10,
                           goal_reward=goal,
                           goal_reward_params=dict(traj_start=True,
                                                   **velocity_profile),
                           use_muscles=use_muscles,
                           obs_avg_window=1, act_avg_window=1)
    else:
        raise NotImplementedError("Invalid goal selected, try one of "
                                  "['trajectory', 'com_vel_trajectory', 'vel_profile', 'max_vel']")
    return mdp


def experiment(goal, use_muscles, n_epochs, n_steps, n_episodes_test):
    np.random.seed(1)

    logger = Logger('SAC', results_dir=None)
    logger.strong_line()
    logger.info('Humanoid Experiment, Algorithm: SAC')

    # MDP
    gamma = 0.99
    horizon = 2000
    mdp = create_mdp(gamma, horizon, goal, use_muscles=use_muscles)

    # Agent
    agent = create_SAC_agent(mdp)

    # normalization callback
    normalizer = MinMaxPreprocessor(mdp_info=mdp.info)
    agent.add_preprocessor(normalizer)

    # plotting callback
    plotter = PlotDataset(mdp.info)

    # Algorithm(with normalization and plotting)
    core = Core(agent, mdp, callback_step=plotter)
    dataset = core.evaluate(n_episodes=n_episodes_test, render=True)

    J = np.mean(compute_J(dataset, gamma))
    L = int(np.round(np.mean(compute_episodes_length(dataset))))

    logger.epoch_info(0, J=J, episode_lenght=L)

    # training loop
    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=True)

        J = np.mean(compute_J(dataset, gamma))
        L = int(np.round(np.mean(compute_episodes_length(dataset))))


        logger.epoch_info(n+1, J=J, episode_lenght=L)

    logger.info('Press a button to visualize humanoid')
    input()
    core.evaluate(n_episodes=10, render=True)


if __name__ == '__main__':
    goal = ["trajectory", "com_vel_trajectory", "vel_profile", "max_vel"]
    experiment(goal=goal[0], use_muscles=True,
               n_epochs=250, n_steps=10000, n_episodes_test=10)