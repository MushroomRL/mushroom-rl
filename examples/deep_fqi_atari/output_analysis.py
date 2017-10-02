import argparse
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from examples.deep_fqi_atari.deep_fqi_atari import Sobel
from examples.deep_fqi_atari.extractor import Extractor
from mushroom.approximators.action_regressor import Regressor
from mushroom.environments import Atari
from mushroom.utils.preprocessor import Binarizer, Scaler
from mushroom.utils.replay_memory import ReplayMemory

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--load-path", type=str)
parser.add_argument("--load-dataset", action='store_true')
parser.add_argument("--game", type=str, default='BreakoutDeterministic-v4')
parser.add_argument("--binarizer-threshold", type=float, default=.1)
parser.add_argument("--n-features", type=int, default=25)
parser.add_argument("--history-length", type=int, default=4)
parser.add_argument("--sobel", action='store_true')
parser.add_argument("--predict-next-frame", action='store_true')
parser.add_argument("--predict-reward", action='store_true')
parser.add_argument("--predict-absorbing", action='store_true')
args = parser.parse_args()

# MDP
mdp = Atari(args.game, 84, 84)
mdp.reset()

extractor_params = dict(folder_name=None,
                        n_actions=mdp.action_space.n,
                        optimizer={'name': 'adam',
                                   'lr': 1,
                                   'decay': 1},
                        width=84,
                        height=84,
                        n_features=args.n_features,
                        history_length=args.history_length,
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

path = args.load_path + '/' + extractor.model._scope_name + '/' +\
       extractor.model._scope_name
restorer = tf.train.import_meta_graph(path + '.meta')
restorer.restore(extractor.model._session, path)
extractor.model._restore_collection()

# Predictions
n_samples = 100
if not args.load_dataset:
    state = np.ones((n_samples, 84, 84, args.history_length))
    action = np.ones((n_samples, 1))
    reward = np.ones((n_samples, 1))
    absorbing = np.ones((n_samples, 1))
    if args.predict_next_frame:
        next_state = np.ones((n_samples, 84, 84))
    else:
        next_state = np.ones((n_samples, 84, 84, args.history_length))
    if args.predict_next_frame:
        for i in xrange(state.shape[0]):
            for j in xrange(4):
                state[i, ..., j], _, _, _ = mdp.step(
                    np.random.randint(mdp.action_space.n))
            a = np.random.randint(mdp.action_space.n)
            next_state[i], r, ab, _ = mdp.step(a)
            action[i] = a
            reward[i] = r
            absorbing[i] = ab
    else:
        for i in xrange(state.shape[0]):
            for j in xrange(args.history_length - 1):
                state[i, ..., j], _, _, _ = mdp.step(
                    np.random.randint(mdp.action_space.n))
                next_state[i, ..., j], _, _, _ = mdp.step(
                    np.random.randint(mdp.action_space.n))
            state[i, ..., -1], _, _, _ = mdp.step(
                np.random.randint(mdp.action_space.n))
            a = np.random.randint(mdp.action_space.n)
            next_state[i, ..., -1], r, ab, _ = mdp.step(a)
            action[i] = a
            reward[i] = r
            absorbing[i] = ab
else:
    dataset = np.load(args.load_path + '/dataset.npy')
    replay_memory = ReplayMemory(len(dataset) + 1, args.history_length)
    mdp_info = dict(observation_space=mdp.observation_space,
                    action_space=mdp.action_space)
    replay_memory.initialize(mdp_info)
    replay_memory.add(dataset)
    state, action, reward, next_state, absorbing, _ = replay_memory.get(
        n_samples)

if args.predict_next_frame:
    extr_input = [state, action]
else:
    extr_input = [state]
reward = np.clip(reward, -1, 1)
for p in preprocessors:
    state = p(state)
    next_state = p(next_state)
y = [next_state, reward, absorbing]
reconstructions = extractor.predict(extr_input, reconstruction=True)[0]
stats = extractor.model.get_stats(extr_input, y)
for key, value in stats.iteritems():
    print('%s: %f' % (key, value))

idxs = list()
for i in xrange(mdp.action_space.n):
    idxs.append(np.argwhere(action == i).ravel()[0])
for idx in idxs:
    plt.figure()
    if args.predict_next_frame:
        for i in xrange(args.history_length):
            plt.subplot(1, args.history_length + 1, i + 1)
            plt.imshow(state[idx, ..., i])
        plt.subplot(1, args.history_length + 1, args.history_length + 1)
        plt.imshow(reconstructions[idx])
    else:
        for i in xrange(args.history_length):
            plt.subplot(2, args.history_length, i + 1)
            plt.imshow(state[idx, ..., i])
            plt.subplot(2, args.history_length, i + args.history_length + 1)
            plt.imshow(reconstructions[idx, ..., i])

plt.show()
