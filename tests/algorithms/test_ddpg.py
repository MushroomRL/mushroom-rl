import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom.algorithms.actor_critic import DDPG, TD3
from mushroom.core import Core
from mushroom.environments.gym_env import Gym
from mushroom.policy import OrnsteinUhlenbeckPolicy


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


def learn(alg):
    mdp = Gym('Pendulum-v0', 200, .99)
    mdp.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

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
                        output_shape=mdp.info.action_space.shape,
                        use_cuda=False)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': .001}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': .001}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         use_cuda=False)

    # Agent
    agent = alg(mdp.info, policy_class, policy_params,
                actor_params, actor_optimizer, critic_params, batch_size,
                initial_replay_size, max_replay_size, tau)

    # Algorithm
    core = Core(agent, mdp)

    core.learn(n_episodes=10, n_episodes_per_fit=5)

    return agent.policy


def test_ddpg():
    policy = learn(DDPG)
    w = policy.get_weights()
    w_test = np.array([-0.28865, -0.7487735, -0.5533644, -0.34702766])

    assert np.allclose(w, w_test)


def test_td3():
    policy = learn(TD3)
    w = policy.get_weights()
    w_test = np.array([1.7005192, -0.73382795, 1.2999079, -0.26730126])

    assert np.allclose(w, w_test)
