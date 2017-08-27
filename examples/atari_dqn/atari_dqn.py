import argparse
import os
import numpy as np

from PyPi.algorithms.dqn import DQN
from PyPi.approximators import Regressor
from PyPi.core.core import Core
from PyPi.environments import *
from PyPi.policy import EpsGreedy
from PyPi.utils.dataset import compute_scores
from PyPi.utils.parameters import LinearDecayParameter, Parameter
from PyPi.utils.preprocessor import Scaler
from convnet import ConvNet
from rmspropgraves import RMSpropGraves

# Disable tf cpp warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def print_epoch(epoch):
    print '################################################################'
    print 'epoch: ', epoch
    print '----------------------------------------------------------------'


def experiment():
    np.random.seed()

    parser = argparse.ArgumentParser()

    arg_game = parser.add_argument_group('Game')
    arg_game.add_argument("--name",
                          type=str,
                          default='BreakoutDeterministic-v4')
    arg_game.add_argument("--screen-width", type=int, default=84)
    arg_game.add_argument("--screen-height", type=int, default=84)

    arg_mem = parser.add_argument_group('Replay Memory')
    arg_mem.add_argument("--initial-replay-size", type=int, default=50000)
    arg_mem.add_argument("--max-replay-size", type=int, default=100000)

    arg_net = parser.add_argument_group('Deep Q-Network')
    arg_net.add_argument("--optimizer",
                         choices=['adam', 'rmsprop', RMSpropGraves()],
                         default=RMSpropGraves())

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--batch-size", type=int, default=32)
    arg_alg.add_argument("--history-length", type=int, default=4)
    arg_alg.add_argument("--target-update-frequency", type=int, default=10000)
    arg_alg.add_argument("--evaluation-frequency", type=int, default=250000)
    arg_alg.add_argument("--train-frequency", type=int, default=4)
    arg_alg.add_argument("--fit-steps", type=int, default=1)
    arg_alg.add_argument("--max-steps", type=int, default=50000000)
    arg_alg.add_argument("--final-exploration-frame", type=int, default=1000000)
    arg_alg.add_argument("--initial-exploration-rate", type=float, default=1)
    arg_alg.add_argument("--final-exploration-rate", type=float, default=.1)
    arg_alg.add_argument("--test-exploration-rate", type=float, default=.05)
    arg_alg.add_argument("--test-samples", type=int, default=125000)
    arg_alg.add_argument("--max-no-op-actions", type=int, default=30)
    arg_alg.add_argument("--no-op-action-value", type=int, default=0)

    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--load-path', type=str)
    arg_utils.add_argument('--save-path', type=str)
    arg_utils.add_argument('--render', action='store_true')
    arg_utils.add_argument('--quiet', action='store_true')
    arg_utils.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    # DQN settings
    if args.debug:
        initial_replay_size = 50
        max_replay_size = 500
        train_frequency = 5
        target_update_frequency = 10
        test_samples = 20
        evaluation_frequency = 10
        max_steps = 1000
    else:
        initial_replay_size = args.initial_replay_size
        max_replay_size = args.max_replay_size
        train_frequency = args.train_frequency
        target_update_frequency = args.target_update_frequency
        test_samples = args.test_samples
        evaluation_frequency = args.evaluation_frequency
        max_steps = args.max_steps

    # MDP train
    mdp = Atari(args.name, args.screen_width, args.screen_height,
                ends_at_life=True)

    # Policy
    epsilon = LinearDecayParameter(value=args.initial_exploration_rate,
                                   min_value=args.final_exploration_rate,
                                   n=args.final_exploration_frame)
    epsilon_test = Parameter(value=args.test_exploration_rate)
    epsilon_random = Parameter(value=1)
    pi = EpsGreedy(epsilon=epsilon_random,
                   observation_space=mdp.observation_space,
                   action_space=mdp.action_space)

    # Approximator
    approximator_params = dict(n_actions=mdp.action_space.n,
                               optimizer=args.optimizer,
                               width=args.screen_width,
                               height=args.screen_height,
                               history_length=args.history_length)
    approximator = Regressor(ConvNet,
                             preprocessor=[Scaler(mdp.observation_space.high)],
                             **approximator_params)

    # target approximator
    target_approximator = Regressor(
        ConvNet,
        preprocessor=[Scaler(mdp.observation_space.high)],
        **approximator_params)
    target_approximator.model.set_weights(approximator.model.get_weights())

    # Agent
    algorithm_params = dict(
        batch_size=args.batch_size,
        target_approximator=target_approximator,
        initial_replay_size=initial_replay_size,
        max_replay_size=max_replay_size,
        history_length=args.history_length,
        train_frequency=train_frequency,
        target_update_frequency=target_update_frequency,
        max_no_op_actions=args.max_no_op_actions,
        no_op_action_value=args.no_op_action_value
    )
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = DQN(approximator, pi, **agent_params)

    # Algorithm
    core = Core(agent, mdp)
    core_test = Core(agent, mdp)

    # DQN

    if args.load_path:
        # evaluate initial policy
        pi.set_epsilon(epsilon_test)
        mdp.set_episode_end(ends_at_life=False)
        approximator.model.load_weights(args.load_path)
        dataset = core_test.evaluate(how_many=test_samples,
                                     iterate_over='samples',
                                     render=args.render,
                                     quiet=args.quiet)
        score = compute_scores(dataset)
        print('min_reward: %f, max_reward: %f, mean_reward: %f,'
              ' games_completed: %d' % score)
    else:
        # fill replay memory with random dataset
        print_epoch(0)
        core.learn(n_iterations=1, how_many=initial_replay_size,
                   n_fit_steps=0, iterate_over='samples', quiet=args.quiet)

        # evaluate initial policy
        pi.set_epsilon(epsilon_test)
        mdp.set_episode_end(ends_at_life=False)
        dataset = core_test.evaluate(how_many=test_samples,
                                     iterate_over='samples',
                                     render=args.render,
                                     quiet=args.quiet)
        score = compute_scores(dataset)
        print('min_reward: %f, max_reward: %f, mean_reward: %f,'
              ' games_completed: %d' % score)
        for i in xrange(max_steps - evaluation_frequency):
            print_epoch(i + 1)
            print '- Learning:'
            # learning step
            pi.set_epsilon(epsilon)
            mdp.set_episode_end(ends_at_life=True)
            core.learn(n_iterations=evaluation_frequency / train_frequency,
                       how_many=train_frequency,
                       n_fit_steps=args.fit_steps,
                       iterate_over='samples',
                       quiet=args.quiet)

            if args.save_path:
                approximator.model.save_weights(args.save_path)

            print '- Evaluation:'
            # evaluation step
            pi.set_epsilon(epsilon_test)
            mdp.set_episode_end(ends_at_life=False)
            core_test.reset()
            dataset = core_test.evaluate(how_many=test_samples,
                                         iterate_over='samples',
                                         render=args.render,
                                         quiet=args.quiet)
            score = compute_scores(dataset)
            print('min_reward: %f, max_reward: %f, mean_reward: %f,'
                  ' games_completed: %d' % score)

if __name__ == '__main__':
    experiment()
