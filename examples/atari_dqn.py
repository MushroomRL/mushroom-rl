import argparse
import datetime
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.value import AveragedDQN, CategoricalDQN, DQN, DoubleDQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.core import Core
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_metrics
from mushroom_rl.utils.parameters import LinearParameter, Parameter
from mushroom_rl.utils.replay_memory import PrioritizedReplayMemory

"""
This script runs Atari experiments with DQN, and some of its variants, as
presented in:
"Human-Level Control Through Deep Reinforcement Learning". Mnih V. et al.. 2015.

"""


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


class FeatureNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[0]

        self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=4)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self._h4 = nn.Linear(3136, Network.n_features)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, state, action=None):
        h = F.relu(self._h1(state.float() / 255.))
        h = F.relu(self._h2(h))
        h = F.relu(self._h3(h))
        h = F.relu(self._h4(h.view(-1, 3136)))

        return h


def print_epoch(epoch):
    print('################################################################')
    print('Epoch: ', epoch)
    print('----------------------------------------------------------------')


def get_stats(dataset):
    score = compute_metrics(dataset)
    print(('min_reward: %f, max_reward: %f, mean_reward: %f,'
          ' games_completed: %d' % score))

    return score


def experiment():
    np.random.seed()

    # Argument parser
    parser = argparse.ArgumentParser()

    arg_game = parser.add_argument_group('Game')
    arg_game.add_argument("--name",
                          type=str,
                          default='BreakoutDeterministic-v4',
                          help='Gym ID of the Atari game.')
    arg_game.add_argument("--screen-width", type=int, default=84,
                          help='Width of the game screen.')
    arg_game.add_argument("--screen-height", type=int, default=84,
                          help='Height of the game screen.')

    arg_mem = parser.add_argument_group('Replay Memory')
    arg_mem.add_argument("--initial-replay-size", type=int, default=50000,
                         help='Initial size of the replay memory.')
    arg_mem.add_argument("--max-replay-size", type=int, default=500000,
                         help='Max size of the replay memory.')
    arg_mem.add_argument("--prioritized", action='store_true',
                         help='Whether to use prioritized memory or not.')

    arg_net = parser.add_argument_group('Deep Q-Network')
    arg_net.add_argument("--optimizer",
                         choices=['adadelta',
                                  'adam',
                                  'rmsprop',
                                  'rmspropcentered'],
                         default='rmsprop',
                         help='Name of the optimizer to use.')
    arg_net.add_argument("--learning-rate", type=float, default=.00025,
                         help='Learning rate value of the optimizer.')
    arg_net.add_argument("--decay", type=float, default=.95,
                         help='Discount factor for the history coming from the'
                              'gradient momentum in rmspropcentered and'
                              'rmsprop')
    arg_net.add_argument("--epsilon", type=float, default=1e-8,
                         help='Epsilon term used in rmspropcentered and'
                              'rmsprop')

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--algorithm", choices=['dqn', 'ddqn', 'adqn', 'cdqn'],
                         default='dqn',
                         help='Name of the algorithm. dqn is for standard'
                              'DQN, ddqn is for Double DQN and adqn is for'
                              'Averaged DQN.')
    arg_alg.add_argument("--n-approximators", type=int, default=1,
                         help="Number of approximators used in the ensemble for"
                              "Averaged DQN.")
    arg_alg.add_argument("--batch-size", type=int, default=32,
                         help='Batch size for each fit of the network.')
    arg_alg.add_argument("--history-length", type=int, default=4,
                         help='Number of frames composing a state.')
    arg_alg.add_argument("--target-update-frequency", type=int, default=10000,
                         help='Number of collected samples before each update'
                              'of the target network.')
    arg_alg.add_argument("--evaluation-frequency", type=int, default=250000,
                         help='Number of collected samples before each'
                              'evaluation. An epoch ends after this number of'
                              'steps')
    arg_alg.add_argument("--train-frequency", type=int, default=4,
                         help='Number of collected samples before each fit of'
                              'the neural network.')
    arg_alg.add_argument("--max-steps", type=int, default=50000000,
                         help='Total number of collected samples.')
    arg_alg.add_argument("--final-exploration-frame", type=int, default=1000000,
                         help='Number of collected samples until the exploration'
                              'rate stops decreasing.')
    arg_alg.add_argument("--initial-exploration-rate", type=float, default=1.,
                         help='Initial value of the exploration rate.')
    arg_alg.add_argument("--final-exploration-rate", type=float, default=.1,
                         help='Final value of the exploration rate. When it'
                              'reaches this values, it stays constant.')
    arg_alg.add_argument("--test-exploration-rate", type=float, default=.05,
                         help='Exploration rate used during evaluation.')
    arg_alg.add_argument("--test-samples", type=int, default=125000,
                         help='Number of collected samples for each'
                              'evaluation.')
    arg_alg.add_argument("--max-no-op-actions", type=int, default=30,
                         help='Maximum number of no-op actions performed at the'
                              'beginning of the episodes.')
    arg_alg.add_argument("--n-atoms", type=int, default=51,
                         help='Number of atoms for Categorical DQN.')
    arg_alg.add_argument("--v-min", type=int, default=-10,
                         help='Minimum action-value for Categorical DQN.')
    arg_alg.add_argument("--v-max", type=int, default=10,
                         help='Maximum action-value for Categorical DQN.')

    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--use-cuda', action='store_true',
                           help='Flag specifying whether to use the GPU.')
    arg_utils.add_argument('--save', action='store_true',
                           help='Flag specifying whether to save the model.')
    arg_utils.add_argument('--load-path', type=str,
                           help='Path of the model to be loaded.')
    arg_utils.add_argument('--render', action='store_true',
                           help='Flag specifying whether to render the game.')
    arg_utils.add_argument('--quiet', action='store_true',
                           help='Flag specifying whether to hide the progress'
                                'bar.')
    arg_utils.add_argument('--debug', action='store_true',
                           help='Flag specifying whether the script has to be'
                                'run in debug mode.')

    args = parser.parse_args()

    scores = list()

    optimizer = dict()
    if args.optimizer == 'adam':
        optimizer['class'] = optim.Adam
        optimizer['params'] = dict(lr=args.learning_rate,
                                   eps=args.epsilon)
    elif args.optimizer == 'adadelta':
        optimizer['class'] = optim.Adadelta
        optimizer['params'] = dict(lr=args.learning_rate,
                                   eps=args.epsilon)
    elif args.optimizer == 'rmsprop':
        optimizer['class'] = optim.RMSprop
        optimizer['params'] = dict(lr=args.learning_rate,
                                   alpha=args.decay,
                                   eps=args.epsilon)
    elif args.optimizer == 'rmspropcentered':
        optimizer['class'] = optim.RMSprop
        optimizer['params'] = dict(lr=args.learning_rate,
                                   alpha=args.decay,
                                   eps=args.epsilon,
                                   centered=True)
    else:
        raise ValueError

    # Summary folder
    folder_name = './logs/atari_' + args.algorithm + '_' + args.name +\
        '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    pathlib.Path(folder_name).mkdir(parents=True)

    # Settings
    if args.debug:
        initial_replay_size = 50
        max_replay_size = 500
        train_frequency = 5
        target_update_frequency = 10
        test_samples = 20
        evaluation_frequency = 50
        max_steps = 1000
    else:
        initial_replay_size = args.initial_replay_size
        max_replay_size = args.max_replay_size
        train_frequency = args.train_frequency
        target_update_frequency = args.target_update_frequency
        test_samples = args.test_samples
        evaluation_frequency = args.evaluation_frequency
        max_steps = args.max_steps

    # MDP
    mdp = Atari(args.name, args.screen_width, args.screen_height,
                ends_at_life=True, history_length=args.history_length,
                max_no_op_actions=args.max_no_op_actions)

    if args.load_path:
        # Agent
        agent = DQN.load(args.load_path)
        epsilon_test = Parameter(value=args.test_exploration_rate)
        agent.policy.set_epsilon(epsilon_test)

        # Algorithm
        core_test = Core(agent, mdp)

        # Evaluate model
        dataset = core_test.evaluate(n_steps=args.test_samples,
                                     render=args.render,
                                     quiet=args.quiet)
        get_stats(dataset)

    else:
        # Policy
        epsilon = LinearParameter(value=args.initial_exploration_rate,
                                  threshold_value=args.final_exploration_rate,
                                  n=args.final_exploration_frame)
        epsilon_test = Parameter(value=args.test_exploration_rate)
        epsilon_random = Parameter(value=1)
        pi = EpsGreedy(epsilon=epsilon_random)

        class CategoricalLoss(nn.Module):
            def forward(self, input, target):
                input = input.clamp(1e-5)

                return -torch.sum(target * torch.log(input))

        # Approximator
        approximator_params = dict(
            network=Network if args.algorithm != 'cdqn' else FeatureNetwork,
            input_shape=mdp.info.observation_space.shape,
            output_shape=(mdp.info.action_space.n,),
            n_actions=mdp.info.action_space.n,
            n_features=Network.n_features,
            optimizer=optimizer,
            loss=F.smooth_l1_loss if args.algorithm != 'cdqn' else CategoricalLoss(),
            use_cuda=args.use_cuda
        )

        approximator = TorchApproximator

        if args.prioritized:
            replay_memory = PrioritizedReplayMemory(
                initial_replay_size, max_replay_size, alpha=.6,
                beta=LinearParameter(.4, threshold_value=1,
                                     n=max_steps // train_frequency)
            )
        else:
            replay_memory = None

        # Agent
        algorithm_params = dict(
            batch_size=args.batch_size,
            n_approximators=args.n_approximators,
            target_update_frequency=target_update_frequency // train_frequency,
            replay_memory=replay_memory,
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size
        )

        if args.algorithm == 'dqn':
            agent = DQN(mdp.info, pi, approximator,
                        approximator_params=approximator_params,
                        **algorithm_params)
        elif args.algorithm == 'ddqn':
            agent = DoubleDQN(mdp.info, pi, approximator,
                              approximator_params=approximator_params,
                              **algorithm_params)
        elif args.algorithm == 'adqn':
            agent = AveragedDQN(mdp.info, pi, approximator,
                                approximator_params=approximator_params,
                                **algorithm_params)
        elif args.algorithm == 'cdqn':
            agent = CategoricalDQN(mdp.info, pi,
                                   approximator_params=approximator_params,
                                   n_atoms=args.n_atoms, v_min=args.v_min,
                                   v_max=args.v_max, **algorithm_params)

        # Algorithm
        core = Core(agent, mdp)

        # RUN

        # Fill replay memory with random dataset
        print_epoch(0)
        core.learn(n_steps=initial_replay_size,
                   n_steps_per_fit=initial_replay_size, quiet=args.quiet)

        if args.save:
            agent.save(folder_name + '/agent_0.msh')

        # Evaluate initial policy
        pi.set_epsilon(epsilon_test)
        mdp.set_episode_end(False)
        dataset = core.evaluate(n_steps=test_samples, render=args.render,
                                quiet=args.quiet)
        scores.append(get_stats(dataset))

        np.save(folder_name + '/scores.npy', scores)
        for n_epoch in range(1, max_steps // evaluation_frequency + 1):
            print_epoch(n_epoch)
            print('- Learning:')
            # learning step
            pi.set_epsilon(epsilon)
            mdp.set_episode_end(True)
            core.learn(n_steps=evaluation_frequency,
                       n_steps_per_fit=train_frequency, quiet=args.quiet)

            if args.save:
                agent.save(folder_name + '/agent_' + str(n_epoch) + '.msh')

            print('- Evaluation:')
            # evaluation step
            pi.set_epsilon(epsilon_test)
            mdp.set_episode_end(False)
            dataset = core.evaluate(n_steps=test_samples, render=args.render,
                                    quiet=args.quiet)
            scores.append(get_stats(dataset))

            np.save(folder_name + '/scores.npy', scores)

    return scores


if __name__ == '__main__':
    experiment()
