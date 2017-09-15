import argparse
import datetime
import glob
import os

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import tensorflow as tf
from tqdm import tqdm

from mushroom.algorithms.batch_td import DeepFQI
from mushroom.approximators import *
from mushroom.core.core import Core
from mushroom.environments import Atari
from mushroom.policy import EpsGreedy
from mushroom.utils.dataset import compute_scores
from mushroom.utils.parameters import Parameter
from mushroom.utils.preprocessor import Binarizer, Scaler
from mushroom.utils.replay_memory import ReplayMemory
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
    arg_game.add_argument("--binarizer-threshold", type=float, default=.1,
                          help='Threshold value to use to binarize images.')

    arg_net = parser.add_argument_group('Extractor')
    arg_net.add_argument("--optimizer",
                         choices=['adadelta',
                                  'adam',
                                  'rmsprop',
                                  'rmspropcentered'],
                         default='adam',
                         help='Name of the optimizer to use to learn.')
    arg_net.add_argument("--learning-rate", type=float, default=.00025,
                         help='Learning rate value of the optimizer. Only works'
                              'for rmsprop and rmspropcentered')
    arg_net.add_argument("--decay", type=float, default=.95,
                         help='Discount factor for the history coming from the'
                              'gradient momentum in rmsprop.')
    arg_net.add_argument("--reg-coeff", type=float, default=1e-5)

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--initial-exploration-rate", type=float, default=1.)
    arg_alg.add_argument("--n-epochs", type=int, default=25)
    arg_alg.add_argument("--n-iterations", type=int, default=10)
    arg_alg.add_argument("--fqi-steps", type=int, default=1000)
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
    arg_utils.add_argument('--load-path-dataset', type=str)
    arg_utils.add_argument('--save-dataset', action='store_true')
    arg_utils.add_argument('--load-path-extractor', type=str)
    arg_utils.add_argument('--save-extractor', action='store_true',
                           help='Flag specifying whether to save the feature'
                                'extractor.')
    arg_utils.add_argument('--load-path-features', type=str)
    arg_utils.add_argument('--save-features', action='store_true')
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
    if args.load_path_extractor and args.load_path_dataset:
        folder_name = args.load_path_extractor
    elif args.load_path_dataset:
        folder_name = args.load_path_dataset
        path = glob.glob(folder_name + '/deep_fqi_extractor*/*')
        for f in path:
            os.remove(f)
    elif args.load_path_extractor:
        folder_name = args.load_path_extractor
    else:
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
                            history_length=args.history_length,
                            reg_coeff=args.reg_coeff)
    extractor = ActionRegressor(Extractor,
                                discrete_actions=mdp.action_space.n,
                                input_preprocessor=[
                                    Scaler(mdp.observation_space.high),
                                    Binarizer(args.binarizer_threshold)],
                                output_preprocessor=[
                                    Scaler(mdp.observation_space.high),
                                    Binarizer(args.binarizer_threshold)],
                                **extractor_params)

    n_features = extractor.models[0].model.n_features

    approximator_params = dict()
    approximator = Regressor(ExtraTreesRegressor,
                             **approximator_params)

    # Agent
    algorithm_params = dict(
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
    core = Core(agent, mdp)

    # Learn
    for k in xrange(args.n_iterations):
        print('Iteration %d' % k)
        if args.load_path_dataset:
            dataset = np.load(
                args.load_path_dataset + '/dataset.npy')[:args.dataset_size]
        else:
            dataset = core.evaluate(how_many=args.dataset_size,
                                    iterate_over='samples',
                                    quiet=args.quiet)
            if args.save_dataset:
                np.save(folder_name + '/dataset.npy', dataset)

        replay_memory = ReplayMemory(args.dataset_size, args.history_length)
        mdp_info = dict(observation_space=mdp.observation_space,
                        action_space=mdp.action_space)
        replay_memory.initialize(mdp_info)
        replay_memory.add(dataset)

        if not args.load_path_extractor:
            for i, m in enumerate(extractor.models):
                print('Fitting model %d' % i)
                best_loss = np.inf
                for e in xrange(args.n_epochs):
                    idxs = np.argwhere(
                        replay_memory._actions.ravel() == i).ravel()
                    rm_generator = replay_memory.generator(args.batch_size,
                                                           idxs)
                    n_batches = int(
                        np.ceil(idxs.size / float(args.batch_size)))

                    gen = tqdm(rm_generator, total=n_batches, dynamic_ncols=100,
                               desc='Epoch %d' % e)
                    for batch in gen:
                        m.train_on_batch(batch[0], batch[3])
                        gen.set_postfix(loss=m.model.loss)

                    if args.save_extractor:
                        if best_loss > m.model.loss:
                            best_loss = m.model.loss
                            m.model.save()
        else:
            for i, e in enumerate(extractor.models):
                restorer = tf.train.import_meta_graph(
                    args.load_path_extractor + '/' + e.model._scope_name + '/' +
                    e.model._scope_name + '.meta')
                restorer.restore(e.model._session, args.load_path_extractor +
                                 '/' + e.model._scope_name + '/' +
                                 e.model._scope_name)
                e.model._restore_collection()

        print('Building features...')
        if not args.load_path_features:
            f = np.ones((replay_memory.size, n_features))
            for i, m in enumerate(extractor.models):
                idxs = np.argwhere(replay_memory._actions.ravel() == i).ravel()
                rm_generator = replay_memory.generator(args.batch_size, idxs)
                for j, batch in enumerate(rm_generator):
                    start = j * batch[0].shape[0]
                    stop = start + batch[0].shape[0]
                    f[start:stop] = m.predict(batch[0])

            ff = np.ones((mdp.action_space.n, replay_memory.size, n_features))
            for i, m in enumerate(extractor.models):
                rm_generator = replay_memory.generator(args.batch_size)
                for j, batch in enumerate(rm_generator):
                    start = j * batch[3].shape[0]
                    stop = start + batch[3].shape[0]
                    ff[i, start:stop] = m.predict(batch[3])

            if args.save_features:
                np.save(folder_name + '/f.npy', f)
                np.save(folder_name + '/ff.npy', ff)
        else:
            f = np.load(args.load_path_features + '/f.npy')
            ff = np.load(args.load_path_features + '/ff.npy')

        print('Starting FQI...')
        dataset = [f, replay_memory._actions, replay_memory._rewards, ff,
                   replay_memory._absorbing, replay_memory._last]
        del replay_memory
        agent.fit(dataset=dataset, n_iterations=args.fqi_steps)

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
