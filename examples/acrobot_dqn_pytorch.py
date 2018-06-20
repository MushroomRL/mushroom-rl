import numpy as np

from mushroom.algorithms.value import DQN
from mushroom.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.approximators.parametric.pytorch_network import *
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import Parameter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, params):
        super(Network, self).__init__()

        n_input = params['input_shape'][0]
        n_output = params['output_shape'][0]
        n_features = params['n_features']

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
        features1 = F.relu(self._h1(torch.squeeze(state, 2)))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action))
            return q_acted


def experiment(n_epochs, n_steps, n_steps_test):
    np.random.seed()

    # MDP
    horizon = 200
    gamma = 0.99
    mdp = Gym('Acrobot-v1', horizon, gamma)

    # Policy
    epsilon = Parameter(value=0.01)
    pi = EpsGreedy(epsilon=epsilon)

    # Settings
    initial_replay_size = 100
    max_replay_size = 5000
    target_update_frequency = 100
    batch_size = 200
    n_features = 80
    train_frequency = 1

    # Approximator
    input_shape = mdp.info.observation_space.shape + (1,)
    approximator_params = dict(network=Network,
                               optimizer={'class': optim.Adam,
                                          'params': {'lr': .0001}},
                               loss=F.smooth_l1_loss,
                               n_features=n_features,
                               input_shape=input_shape,
                               output_shape=mdp.info.action_space.size,
                               n_actions=mdp.info.action_space.n)


    # Agent
    agent = DQN(PyTorchApproximator, pi, mdp.info,
                approximator_params=approximator_params,
                batch_size=batch_size,
                n_approximators=1,
                initial_replay_size=initial_replay_size,
                max_replay_size=max_replay_size,
                history_length=1,
                target_update_frequency=target_update_frequency,
                max_no_op_actions=0,
                no_op_action_value=0,
                dtype=np.float32)


    # Algorithm
    core = Core(agent, mdp)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    J = compute_J(dataset, gamma)
    print('J: ', np.mean(J))

    for n in range(n_epochs):
        print('Epoch: ', n)
        core.learn(n_steps=n_steps,
                   n_steps_per_fit=train_frequency)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        J = compute_J(dataset, gamma)
        print('J: ', np.mean(J))

    print('Press a button to visualize acrobot')
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':
    experiment(n_epochs=50, n_steps=1000, n_steps_test=1000)
