import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.value import DQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import AtariNetwork
from mushroom_rl.core import Core
from mushroom_rl.environments import Atari
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import LinearParameter, Parameter


def print_epoch(epoch):
    print('################################################################')
    print('Epoch: ', epoch)
    print('----------------------------------------------------------------')


def get_stats(dataset):
    score = dataset.compute_metrics()
    print(('min_reward: %f, max_reward: %f, mean_reward: %f,'
          ' median_reward: %f, games_completed: %d' % score))

    return score


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
mdp = Atari('ALE/Breakout-v5', width, height)

# Policy
epsilon = LinearParameter(value=1.,
                          threshold_value=.1,
                          n=1000000)
epsilon_test = Parameter(value=.05)
epsilon_random = Parameter(value=1)
pi = EpsGreedy(epsilon=epsilon_random, backend='torch')

# Approximator
input_shape = (history_length,) + mdp.info.observation_space.shape
approximator_params = dict(
    network=AtariNetwork,
    input_shape=input_shape,
    output_shape=(mdp.info.action_space.n,),
    n_actions=mdp.info.action_space.n,
    n_features=AtariNetwork.n_features,
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
    max_replay_size=max_replay_size,
    history_length=history_length
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
dataset = core.evaluate(n_steps=test_samples)
scores = [get_stats(dataset)]

for n_epoch in range(1, max_steps // evaluation_frequency + 1):
    print_epoch(n_epoch)
    print('- Learning:')
    pi.set_epsilon(epsilon)
    core.learn(n_steps=evaluation_frequency,
               n_steps_per_fit=train_frequency)

    print('- Evaluation:')
    pi.set_epsilon(epsilon_test)
    dataset = core.evaluate(n_steps=test_samples)
    scores.append(get_stats(dataset))