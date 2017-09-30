import argparse
import datetime

import joblib
import numpy as np
from scipy.ndimage import sobel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
import tensorflow as tf
from tqdm import tqdm

from mushroom.algorithms.batch_td import DeepFQI
from mushroom.approximators import *
from mushroom.core.core import Core
from mushroom.environments import Atari
from mushroom.policy import EpsGreedy
from mushroom.utils.dataset import compute_scores
from mushroom.utils.ifs import IFS
from mushroom.utils.parameters import Parameter
from mushroom.utils.preprocessor import Binarizer, Filter, Preprocessor, Scaler
from mushroom.utils.replay_memory import ReplayMemory
from mushroom.utils.rfs import RFS
from extractor import Extractor


def get_stats(dataset):
    score = compute_scores(dataset)
    print('min_reward: %f, max_reward: %f, mean_reward: %f,'
          ' games_completed: %d' % score)

    return score


class Sobel(Preprocessor):
    def __init__(self, history_length, mode='reflect'):
        self._history_length = history_length
        self._mode = mode

    def _compute(self, imgs):
        filter_imgs = np.ones(imgs.shape)
        for s in xrange(imgs.shape[0]):
            for h in xrange(self._history_length):
                filter_x = sobel(imgs[s, ..., h], axis=0, mode=self._mode)
                filter_y = sobel(imgs[s, ..., h], axis=1, mode=self._mode)
                filter_imgs[s, ..., h] = np.sqrt(
                    filter_x ** 2 + filter_y ** 2)

        return filter_imgs


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
    arg_net.add_argument("--n-features", type=int, default=25)
    arg_net.add_argument("--reg-coeff", type=float, default=1e-5)
    arg_net.add_argument("--contractive", action='store_true')
    arg_net.add_argument("--sobel", action='store_true')
    arg_net.add_argument("--predict-next-frame", action='store_true')
    arg_net.add_argument("--predict-reward", action='store_true')
    arg_net.add_argument("--predict-absorbing", action='store_true')

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--initial-exploration-rate", type=float, default=1.)
    arg_alg.add_argument("--n-epochs", type=int, default=25)
    arg_alg.add_argument("--n-iterations", type=int, default=10)
    arg_alg.add_argument("--fqi-steps", type=int, default=1000)
    arg_alg.add_argument("--approximator", choices=['linear', 'extra'],
                         default='extra')
    arg_alg.add_argument("--n-estimators", type=int, default=50)
    arg_alg.add_argument("--min-samples-split", type=int, default=5)
    arg_alg.add_argument("--min-samples-leaf", type=int, default=2)
    arg_alg.add_argument("--max-depth", type=int, default=None)
    arg_alg.add_argument("--dataset-size", type=int, default=500000)
    arg_alg.add_argument("--validation-split", type=float, default=.2)
    arg_alg.add_argument("--batch-size", type=int, default=32,
                         help='Batch size for each fit of the network.')
    arg_alg.add_argument("--history-length", type=int, default=4,
                         help='Number of frames composing a state.')
    arg_alg.add_argument("--test-samples", type=int, default=125000,
                         help='Number of steps for each evaluation.')
    arg_alg.add_argument("--evaluation-frequency", type=int, default=50)
    arg_alg.add_argument("--max-no-op-actions", type=int, default=30,
                         help='Maximum number of no-op action performed at the'
                              'beginning of the episodes. The minimum number is'
                              'history_length.')
    arg_alg.add_argument("--no-op-action-value", type=int, default=0,
                         help='Value of the no-op action.')

    arg_rfs = parser.add_argument_group('RFS')
    arg_rfs.add_argument("--rfs", action='store_true')
    arg_rfs.add_argument("--rfs-n-estimators", type=int, default=50)
    arg_rfs.add_argument("--rfs-min-samples-split", type=int, default=5)
    arg_rfs.add_argument("--rfs-min-samples-leaf", type=int, default=2)
    arg_rfs.add_argument("--rfs-max-depth", type=int, default=None)
    arg_rfs.add_argument('--load-support', action='store_true')
    arg_rfs.add_argument('--save-support', action='store_true')

    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--load-path', type=str)
    arg_utils.add_argument('--load-dataset', action='store_true')
    arg_utils.add_argument('--save-dataset', action='store_true')
    arg_utils.add_argument('--load-extractor', action='store_true')
    arg_utils.add_argument('--save-extractor', action='store_true',
                           help='Flag specifying whether to save the feature'
                                'extractor.')
    arg_utils.add_argument('--load-features', action='store_true')
    arg_utils.add_argument('--save-features', action='store_true')
    arg_utils.add_argument('--load-approximator', action='store_true')
    arg_utils.add_argument('--save-approximator', action='store_true')
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
    if args.load_path:
        folder_name = args.load_path
    else:
        folder_name = './logs/deep_fqi_atari_' +\
                      datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # MDP
    mdp = Atari(args.name, args.screen_width, args.screen_height,
                ends_at_life=True)

    # Policy
    epsilon = Parameter(value=args.initial_exploration_rate)
    pi = EpsGreedy(epsilon=epsilon,
                   observation_space=mdp.observation_space,
                   action_space=mdp.action_space)

    # Feature extractor
    if args.load_extractor:
        extractor_folder_name = None
    else:
        extractor_folder_name = folder_name
    extractor_params = dict(folder_name=extractor_folder_name,
                            n_actions=mdp.action_space.n,
                            optimizer={'name': args.optimizer,
                                       'lr': args.learning_rate,
                                       'decay': args.decay},
                            width=args.screen_width,
                            height=args.screen_height,
                            history_length=args.history_length,
                            n_features=args.n_features,
                            reg_coeff=args.reg_coeff,
                            contractive=args.contractive,
                            predict_next_frame=args.predict_next_frame,
                            predict_reward=args.predict_reward,
                            predict_absorbing=args.predict_absorbing)

    preprocessors = [Scaler(mdp.observation_space.high),
                     Binarizer(args.binarizer_threshold)]
    if args.sobel:
        preprocessors += [Sobel(args.history_length), Binarizer(0, False)]
    if args.predict_next_frame:
        extractor = Regressor(Extractor,
                              discrete_actions=mdp.action_space.n,
                              input_preprocessor=preprocessors,
                              output_preprocessor=preprocessors,
                              **extractor_params)
    else:
        extractor = Regressor(Extractor,
                              input_preprocessor=preprocessors,
                              output_preprocessor=preprocessors,
                              **extractor_params)
    n_features = extractor.model.n_features

    if args.predict_next_frame:
        approximator_class = Regressor
        discrete_actions = None
    else:
        approximator_class = ActionRegressor
        discrete_actions = mdp.action_space.n

    if args.approximator == 'extra':
        approximator_params = dict(n_estimators=args.n_estimators,
                                   min_samples_split=args.min_samples_split,
                                   min_samples_leaf=args.min_samples_leaf,
                                   max_depth=args.max_depth)
        approximator = approximator_class(ExtraTreesRegressor,
                                          discrete_actions=discrete_actions,
                                          **approximator_params)
    elif args.approximator == 'linear':
        approximator_params = dict()
        approximator = approximator_class(LinearRegression,
                                          discrete_actions=discrete_actions,
                                          **approximator_params)
    else:
        raise ValueError

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
        if not args.load_approximator or k > 0:
            if not args.load_dataset or k > 0:
                pi.set_epsilon(Parameter(1))
                mdp.set_episode_end(ends_at_life=True)
                dataset = core.evaluate(how_many=args.dataset_size,
                                        iterate_over='samples',
                                        quiet=args.quiet)
                if args.save_dataset:
                    np.save(folder_name + '/dataset.npy', dataset)
            else:
                dataset = np.load(
                    folder_name + '/dataset.npy')[:args.dataset_size]

            replay_memory = ReplayMemory(len(dataset), args.history_length)
            mdp_info = dict(observation_space=mdp.observation_space,
                            action_space=mdp.action_space)
            replay_memory.initialize(mdp_info)
            replay_memory.add(dataset)

            del dataset

        if not args.load_extractor or k > 0:
            extractor.model._folder_name = folder_name
            print('Fitting extractor...')
            best_loss = np.inf
            idxs = np.arange(replay_memory.size)
            np.random.shuffle(idxs)
            valid_start = int(
                replay_memory.size - replay_memory.size * args.validation_split)
            train_idxs = idxs[:valid_start]
            valid_idxs = idxs[valid_start:]
            for e in xrange(args.n_epochs):
                rm_generator = replay_memory.generator(args.batch_size,
                                                       train_idxs)
                n_batches = int(
                    np.ceil(train_idxs.size / float(args.batch_size)))

                gen = tqdm(rm_generator, total=n_batches, dynamic_ncols=100,
                           desc='Epoch %d' % e)
                for batch in gen:
                    if args.predict_next_frame:
                        extr_input = [batch[0], batch[1]]
                        target = batch[3][..., -1]
                    else:
                        extr_input = [batch[0]]
                        target = batch[0]
                    extractor.train_on_batch(
                        extr_input,
                        target,
                        target_reward=batch[2].reshape(-1, 1),
                        target_absorbing=batch[4].reshape(-1, 1)
                    )
                    gen.set_postfix(loss=extractor.model.loss)

                valid_rm_generator = replay_memory.generator(args.batch_size,
                                                             valid_idxs)
                valid_loss = 0.
                for valid_batch in valid_rm_generator:
                    if args.predict_next_frame:
                        extr_input = [valid_batch[0], valid_batch[1]]
                        target = valid_batch[3][..., -1]
                    else:
                        extr_input = [valid_batch[0]]
                        target = valid_batch[0]
                    for p in preprocessors:
                        extr_input = p(extr_input)
                        target = p(target)
                    valid_loss += extractor.model.get_stats(
                        extr_input, target)['loss'] * valid_batch[0].shape[0]
                valid_loss /= float(valid_idxs.size)
                print('valid_loss=%f' % valid_loss)
                if args.save_extractor:
                    if best_loss > valid_loss:
                        best_loss = valid_loss
                        extractor.model.save()
        else:
            restorer = tf.train.import_meta_graph(
                folder_name + '/' + extractor.model._scope_name + '/' +
                extractor.model._scope_name + '.meta')
            restorer.restore(
                extractor.model._session, folder_name +
                '/' + extractor.model._scope_name + '/' +
                extractor.model._scope_name)
            extractor.model._restore_collection()

        print('Building features...')
        if not args.load_approximator or k > 0:
            if not args.load_features or k > 0:
                f = np.ones((train_idxs.size, n_features))
                actions = np.ones((train_idxs.size, 1))
                rewards = np.ones(train_idxs.size)
                absorbing = np.ones(train_idxs.size)
                last = np.ones(train_idxs.size)
                if args.predict_next_frame:
                    ff = np.ones((mdp.action_space.n, train_idxs.size,
                                  n_features))
                else:
                    ff = np.ones((train_idxs.size, n_features))
                rm_generator = replay_memory.generator(args.batch_size,
                                                       train_idxs)
                for i, batch in enumerate(rm_generator):
                    start = i * args.batch_size
                    stop = start + batch[0].shape[0]
                    sa = [batch[0], batch[1]]
                    f[start:stop] = extractor.predict(sa)[0]
                    actions[start:stop] = batch[1]
                    rewards[start:stop] = batch[2]
                    absorbing[start:stop] = batch[4]
                    last[start:stop] = batch[5]
                    if args.predict_next_frame:
                        for j in xrange(mdp.action_space.n):
                            start = i * args.batch_size
                            stop = start + batch[3].shape[0]
                            sa_n = [batch[3], np.ones(
                                (batch[3].shape[0], 1)) * j]
                            ff[j, start:stop] = extractor.predict(sa_n)[0]
                    else:
                        ss = [batch[3]]
                        ff[start:stop] = extractor.predict(ss)[0]

                del replay_memory

                if args.save_features:
                    np.savez(folder_name + '/feature_dataset.npz',
                             f=f, actions=actions, rewards=rewards, ff=ff,
                             absorbing=absorbing, last=last)
            else:
                files = np.load(folder_name + '/feature_dataset.npz')
                f = files['f']
                actions = files['actions']
                rewards = files['rewards']
                ff = files['ff']
                absorbing = files['absorbing']
                last = files['last']

        if args.rfs:
            if not args.load_support or k > 0:
                print('Starting RFS...')
                ifs_estimator_params = {'n_estimators': args.rfs_n_estimators,
                                        'n_jobs': -1}
                ifs_params = {'estimator': ExtraTreesRegressor(
                    **ifs_estimator_params)}
                ifs = IFS(**ifs_params)
                features_names = np.array(map(str,
                                              np.arange(f.shape[1])) + ['A'])
                rfs_params = {'feature_selector': ifs,
                              'features_names': features_names,
                              'verbose': 1}
                rfs = RFS(**rfs_params)
                rfs.fit(f, actions, ff, rewards)

                support_rfs = np.array(rfs.get_support())
                got_action = support_rfs[-1]  # Action is the last feature
                support_rfs = support_rfs[:-1]  # Remove action
                nb_new_features = np.sum(support_rfs)
                print('Using %s features' % nb_new_features)
                print('Action was%s selected' % ('' if got_action else ' NOT'))

                if args.save_support:
                    np.save(folder_name + '/support.npy', support_rfs)
            else:
                support_rfs = np.load(folder_name + '/support.npy')

            approximator._input_preprocessor = [Filter(support_rfs)]

        print('Starting FQI...')
        if not args.load_approximator or k > 0:
            dataset = [f, actions, rewards, ff, absorbing, last]

            pi.set_epsilon(Parameter(.05))
            mdp.set_episode_end(ends_at_life=False)
            y = None
            max_mean_score = 0
            for i in tqdm(xrange(args.fqi_steps), dynamic_ncols=True,
                          disable=args.quiet, leave=False):
                y = agent._partial_fit(dataset, y)
                if i % args.evaluation_frequency == 0 and i > 0:
                    print('- Evaluation')
                    # evaluation step
                    core.reset()
                    results = core.evaluate(how_many=args.test_samples,
                                            iterate_over='samples',
                                            render=args.render,
                                            quiet=args.quiet)
                    _, _, mean_score, _ = get_stats(results)

                    if args.save_approximator:
                        if mean_score > max_mean_score:
                            if approximator_class == Regressor:
                                joblib.dump(approximator.model,
                                            folder_name + '/approximator.pkl')
                            else:
                                for m_i, m in enumerate(approximator.models):
                                    joblib.dump(
                                        m,
                                        folder_name +
                                        '/approximator_%d.pkl' % m_i)
        else:
            if approximator_class == Regressor:
                approximator.model = joblib.load(
                    folder_name + '/approximator.pkl')
            else:
                for m_i in xrange(len(approximator.models)):
                    approximator.models[m_i] = joblib.load(
                        folder_name + '/approximator_%d.pkl' % m_i)

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
