import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom.algorithms.value import DQN
from mushroom.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.approximators.parametric.pytorch_network import *
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import Parameter, LinearDecayParameter


class Network(nn.Module):
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

    def forward(self, state, action=None):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted


def experiment(n_epochs, n_steps, n_steps_test):
    np.random.seed()

    # MDP
    horizon = 1000
    gamma = 0.99
    gamma_eval = 1.
    mdp = Gym('Acrobot-v1', horizon, gamma)

    # Policy
    epsilon = LinearDecayParameter(value=1., min_value=.01, n=5000)
    epsilon_test = Parameter(value=0.)
    epsilon_random = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon_random)

    # Settings
    initial_replay_size = 500
    max_replay_size = 5000
    target_update_frequency = 100
    batch_size = 200
    n_features = 80
    train_frequency = 1

    # Approximator
    input_shape = mdp.info.observation_space.shape
    approximator_params = dict(network=Network,
                               optimizer={'class': optim.Adam,
                                          'params': {'lr': .001}},
                               loss=F.smooth_l1_loss,
                               n_features=n_features,
                               input_shape=input_shape,
                               output_shape=mdp.info.action_space.size,
                               n_actions=mdp.info.action_space.n)

    # Agent
    agent = DQN(PyTorchApproximator, pi, mdp.info,
                approximator_params=approximator_params, batch_size=batch_size,
                n_approximators=1, initial_replay_size=initial_replay_size,
                max_replay_size=max_replay_size,
                target_update_frequency=target_update_frequency)

    # Algorithm
    core = Core(agent, mdp)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    # RUN
    pi.set_epsilon(epsilon_test)
    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    J = compute_J(dataset, gamma_eval)
    print('J: ', np.mean(J))

    for n in range(n_epochs):
        print('Epoch: ', n)
        pi.set_epsilon(epsilon)
        core.learn(n_steps=n_steps, n_steps_per_fit=train_frequency)
        pi.set_epsilon(epsilon_test)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        J = compute_J(dataset, gamma_eval)
        print('J: ', np.mean(J))

    print('Press a button to visualize acrobot')
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':
    experiment(n_epochs=20, n_steps=1000, n_steps_test=2000)
