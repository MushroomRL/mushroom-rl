import argparse
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from examples.deep_fqi_atari.extractor import Extractor
from mushroom.approximators.action_regressor import Regressor
from mushroom.environments import Atari
from mushroom.utils.preprocessor import Binarizer, Scaler

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--load-path", type=str)
parser.add_argument("--game", type=str, default='BreakoutDeterministic-v4')
parser.add_argument("--binarizer-threshold", type=float, default=.1)
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
                        history_length=4,
                        predict_reward=args.predict_reward,
                        predict_absorbing=args.predict_absorbing)
extractor = Regressor(Extractor,
                      discrete_actions=mdp.action_space.n,
                      input_preprocessor=[
                          Scaler(255.),
                          Binarizer(args.binarizer_threshold)
                      ],
                      output_preprocessor=[
                          Scaler(255.),
                          Binarizer(args.binarizer_threshold)
                      ],
                      **extractor_params)

path =\
    args.load_path + '/' + extractor.model._scope_name
restorer = tf.train.import_meta_graph(path + '.meta')
restorer.restore(extractor.model._session, path)
extractor.model._restore_collection()

# Predictions
n_samples = 500
state = np.ones((n_samples, 84, 84, 4))
action = np.ones((n_samples, 1))
reward = np.ones((n_samples, 1))
absorbing = np.ones((n_samples, 1))
next_state = np.ones((n_samples, 84, 84))

for i in xrange(state.shape[0]):
    for j in xrange(4):
        state[i, ..., j], _, _, _ = mdp.step(
            np.random.randint(mdp.action_space.n))
    a = np.random.randint(mdp.action_space.n)
    next_state[i], r, ab, _ = mdp.step(a)
    action[i] = a
    reward[i] = r
    absorbing[i] = ab

sa = [state, action]
reward = np.clip(reward, -1, 1)
y = [(next_state / 255. >= args.binarizer_threshold).astype(np.float),
     reward,
     absorbing
     ]
predictions = extractor.predict(sa, reconstruction=True)
stats = extractor.model.get_stats(sa, y)
for key, value in stats.iteritems():
    print('%s: %f' % (key, value))

idxs = list()
for i in xrange(mdp.action_space.n):
    idxs.append(np.argwhere(action == i).ravel()[0])
for idx in idxs:
    plt.figure()
    for i in xrange(4):
        plt.subplot(1, 5, i + 1)
        plt.imshow(state[idx, ..., i])
    plt.subplot(1, 5, 5)
    plt.imshow(predictions[idx])

plt.show()
