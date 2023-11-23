import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.value import DQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.core import Core
from mushroom_rl.environments import Atari
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import LinearParameter, Parameter


class Network(nn.Module):
    n_features = 512

    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[0]
        n_output = output_shape[0]

        self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=4)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self._h4 = nn.Linear(3136, self.n_features)
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

    def forward(self, state, action=None):
        h = F.relu(self._h1(state.float() / 255.))
        h = F.relu(self._h2(h))
        h = F.relu(self._h3(h))
        h = F.relu(self._h4(h.view(-1, 3136)))
        q = self._h5(h)

        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))

            return q_acted


def print_epoch(epoch):
    print('################################################################')
    print('Epoch: ', epoch)
    print('----------------------------------------------------------------')


def get_stats(dataset):
    score = dataset.compute_metrics()
    print(('min_reward: %f, max_reward: %f, mean_reward: %f,'
          ' median_reward: %f, games_completed: %d' % score))

    return score


scores = list()

optimizer = dict()
optimizer['class'] = optim.Adam
optimizer['params'] = dict(lr=.00025)

# Settings
width = 84
height = 84
history_length = 4
train_frequency = 4
evaluation_frequency = 250000
target_update_frequency = 10000
initial_replay_size = 50000
max_replay_size = 500000
test_samples = 125000
max_steps = 50000000

# MDP
mdp = Atari('BreakoutDeterministic-v4', width, height, ends_at_life=True,
            history_length=history_length, max_no_op_actions=30)

# Policy
epsilon = LinearParameter(value=1.,
                          threshold_value=.1,
                          n=1000000)
epsilon_test = Parameter(value=.05)
epsilon_random = Parameter(value=1)
pi = EpsGreedy(epsilon=epsilon_random)

# Approximator
input_shape = (history_length, height, width)
approximator_params = dict(
    network=Network,
    input_shape=input_shape,
    output_shape=(mdp.info.action_space.n,),
    n_actions=mdp.info.action_space.n,
    n_features=Network.n_features,
    optimizer=optimizer,
    loss=F.smooth_l1_loss
)

approximator = TorchApproximator

# Agent
algorithm_params = dict(
    batch_size=32,
    target_update_frequency=target_update_frequency // train_frequency,
    replay_memory=None,
    initial_replay_size=initial_replay_size,
    max_replay_size=max_replay_size
)

agent = DQN(mdp.info, pi, approximator,
            approximator_params=approximator_params,
            **algorithm_params)

# Algorithm
core = Core(agent, mdp)

# RUN

# Fill replay memory with random dataset
print_epoch(0)
core.learn(n_steps=initial_replay_size,
           n_steps_per_fit=initial_replay_size)

# Evaluate initial policy
pi.set_epsilon(epsilon_test)
mdp.set_episode_end(False)
dataset = core.evaluate(n_steps=test_samples)
scores.append(get_stats(dataset))

for n_epoch in range(1, max_steps // evaluation_frequency + 1):
    print_epoch(n_epoch)
    print('- Learning:')
    # learning step
    pi.set_epsilon(epsilon)
    mdp.set_episode_end(True)
    core.learn(n_steps=evaluation_frequency,
               n_steps_per_fit=train_frequency)

    print('- Evaluation:')
    # evaluation step
    pi.set_epsilon(epsilon_test)
    mdp.set_episode_end(False)
    dataset = core.evaluate(n_steps=test_samples)
    scores.append(get_stats(dataset))
