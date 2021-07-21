import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.core import Agent
from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core
from mushroom_rl.environments.gym_env import Gym


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


def learn_sac():
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
    warmup_transitions = 10
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
    agent = SAC(mdp.info, actor_mu_params, actor_sigma_params, actor_optimizer,
                critic_params, batch_size, initial_replay_size, max_replay_size,
                warmup_transitions, tau, lr_alpha,
                critic_fit_params=None)

    # Algorithm
    core = Core(agent, mdp)

    core.learn(n_steps=2 * initial_replay_size,
               n_steps_per_fit=initial_replay_size)
    
    return agent

def test_sac():
    policy = learn_sac().policy
    w = policy.get_weights()
    w_test = np.array([ 1.6998193, -0.732528, 1.2986078, -0.26860124,
                        0.5094043, -0.5001421, -0.18989229, -0.30646914])

    assert np.allclose(w, w_test)

def test_sac_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    agent_save = learn_sac()

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)
