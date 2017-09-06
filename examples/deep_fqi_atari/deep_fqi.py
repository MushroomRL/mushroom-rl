import argparse
import datetime

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from mushroom.algorithms import DeepFQI
from mushroom.approximators import *
from mushroom.utils.callbacks import EvaluatePolicy
from mushroom.core.core import Core
from mushroom.environments import Atari
from mushroom.policy import EpsGreedy
from mushroom.utils.parameters import Parameter
from mushroom.utils.preprocessor import Scaler
from mushroom.utils.dataset import compute_scores
from extractor import Extractor


def get_stats(dataset):
    score = compute_scores(dataset)
    print('min_reward: %f, max_reward: %f, mean_reward: %f,'
          ' games_completed: %d' % score)


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

    arg_net = parser.add_argument_group('Extractor')
    arg_net.add_argument("--optimizer",
                         choices=['adadelta',
                                  'adam',
                                  'rmsprop',
                                  'rmspropcentered'],
                         default='rmspropcentered',
                         help='Name of the optimizer to use to learn.')
    arg_net.add_argument("--learning-rate", type=float, default=.00025,
                         help='Learning rate value of the optimizer. Only works'
                              'for rmsprop and rmspropcentered')
    arg_net.add_argument("--decay", type=float, default=.95,
                         help='Discount factor for the history coming from the'
                              'gradient momentum in rmsprop.')

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--initial-exploration-rate", type=float, default=1.)
    arg_alg.add_argument("--n-epochs", type=int, default=3)
    arg_alg.add_argument("--n-iterations", type=int, default=10)
    arg_alg.add_argument("--fit-steps", type=int, default=125000)
    arg_alg.add_argument("--dataset-size", type=int, default=500000)
    arg_alg.add_argument("--batch-size", type=int, default=32,
                         help='Batch size for each fit of the network.')
    arg_alg.add_argument("--history-length", type=int, default=4,
                         help='Number of frames composing a state.')
    arg_alg.add_argument("--test-samples", type=int, default=125000,
                         help='Number of steps for each evaluation.')
    arg_alg.add_argument("--max-no-op-actions", type=int, default=30,
                         help='Maximum number of no-op action performed at the'
                              'beginning of the episodes. The minimum number is'
                              'history_length.')
    arg_alg.add_argument("--no-op-action-value", type=int, default=0,
                         help='Value of the no-op action.')

    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--render', action='store_true',
                           help='Flag specifying whether to render the game.')
    arg_utils.add_argument('--quiet', action='store_true',
                           help='Flag specifying whether to hide the progress'
                                'bar.')
    arg_utils.add_argument('--debug', action='store_true',
                           help='Flag specifying whether the script has to be'
                                'run in debug mode.')

    args = parser.parse_args()

    # Summary folder
    folder_name = './logs/' + datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S')

    # MDP
    mdp = Atari(args.name, args.screen_width, args.screen_height,
                ends_at_life=True)

    # Policy
    epsilon = Parameter(value=args.initial_exploration_rate)
    pi = EpsGreedy(epsilon=epsilon,
                   observation_space=mdp.observation_space,
                   action_space=mdp.action_space)

    # Feature extractor
    extractor_params = dict(folder_name=folder_name,
                            n_actions=mdp.action_space.n,
                            optimizer={'name': args.optimizer,
                                       'lr': args.learning_rate,
                                       'decay': args.decay},
                            width=args.screen_width,
                            height=args.screen_height,
                            history_length=args.history_length)
    extractor = ActionRegressor(Extractor,
                                discrete_actions=mdp.action_space.n,
                                state_preprocessor=[
                                    Scaler(mdp.observation_space.high)],
                                **extractor_params)

    approximator_params = dict()
    approximator = Regressor(ExtraTreesRegressor, **approximator_params)

    # Agent
    algorithm_params = dict(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        predict_next_state=True,
        dataset_size=args.dataset_size,
        extractor=extractor,
        history_length=args.history_length,
        max_no_op_actions=args.max_no_op_actions,
        no_op_action_value=args.no_op_action_value
    )
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}

    agent = DeepFQI(approximator, pi, mdp.gamma, **agent_params)

    # Core
    evaluate_policy = EvaluatePolicy(how_many=args.test_samples,
                                     iterate_over='samples',
                                     render=args.render,
                                     quiet=args.quiet)
    core = Core(agent, mdp, callbacks=[evaluate_policy])

    # Learn
    core.learn(n_iterations=args.n_iterations, how_many=args.dataset_size,
               n_fit_steps=args.fit_steps, iterate_over='samples',
               quiet=args.quiet)

    print '- Evaluation:'
    # evaluation step
    pi.set_epsilon(Parameter(.05))
    mdp.set_episode_end(ends_at_life=False)
    core.reset()
    dataset = core.evaluate(how_many=args.test_samples,
                            iterate_over='samples',
                            render=args.render,
                            quiet=args.quiet)
    get_stats(dataset)

if __name__ == '__main__':
    experiment()
