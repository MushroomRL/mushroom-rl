import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.value import DQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter


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


def experiment():

    logger = Logger(log_name='get_env_info', results_dir=None)
    optimizer = {'class': optim.Adam,
                 'params': dict(lr=1e-4, eps=1e-8)
    }

    # Settings
    initial_replay_size = 50
    max_replay_size = 500
    train_frequency = 5
    target_update_frequency = 10
    test_samples = 20

    # MDP
    mdp = Atari('BreakoutDeterministic-v4', 84, 84,
                ends_at_life=True, history_length=4,
                max_no_op_actions=30)

    # Policy
    epsilon_test = Parameter(value=.05)
    epsilon_random = Parameter(value=1)
    pi = EpsGreedy(epsilon=epsilon_random)

    # Approximator
    approximator_params = dict(
        network=Network,
        input_shape=mdp.info.observation_space.shape,
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
    logger.info('Running a learn run')
    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size, quiet=True)

    # Evaluate initial policy
    pi.set_epsilon(epsilon_test)
    mdp.set_episode_end(False)
    logger.info('Evaluate, without getting the env info dictionary')
    dataset = core.evaluate(n_steps=test_samples, render=False, quiet=True, get_env_info=False)

    logger.info(f'Dataset length {len(dataset)}')

    logger.info('Evaluate, returning also the env info dictionary')
    dataset, dataset_info = core.evaluate(n_steps=test_samples, render=False, quiet=True, get_env_info=True)

    logger.info(f'Dataset length {len(dataset)}')
    logger.info(f'Dataset length {len(dataset_info)}')

    for i, step_info in enumerate(dataset_info):
        logger.epoch_info(i, **step_info)


if __name__ == '__main__':
    experiment()
