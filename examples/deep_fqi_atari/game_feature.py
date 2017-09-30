#!/usr/bin/env python
from __future__ import print_function

import argparse
import datetime
import gym

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from PIL import Image

from examples.atari_dqn.convnet import ConvNet
from examples.deep_fqi_atari.deep_fqi_atari import Sobel
from examples.deep_fqi_atari.extractor import Extractor
from mushroom.approximators.action_regressor import Regressor
from mushroom.utils.preprocessor import Scaler, Binarizer
from mushroom.utils.replay_memory import Buffer

#
# Test yourself as a learning agent! Pass environment name as a command-line argument.
#
# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--load-path", type=str)
parser.add_argument("--game", type=str, default='BreakoutDeterministic-v4')
parser.add_argument("--binarizer_threshold", type=float, default=.1)
parser.add_argument("--n-features", type=int, default=25)
parser.add_argument("--history-length", type=int, default=4)
parser.add_argument("--dqn", action='store_true')
parser.add_argument("--sobel", action='store_true')
parser.add_argument("--predict-next-frame", action='store_true')
parser.add_argument("--predict-reward", action='store_true')
parser.add_argument("--predict-absorbing", action='store_true')
args = parser.parse_args()

env = gym.make(args.game)

# Feature extractor
folder_name = './' + datetime.datetime.now().strftime(
    '%Y-%m-%d_%H-%M-%S')

if not args.dqn:
    extractor_params = dict(folder_name=None,
                            n_actions=env.action_space.n,
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

    preprocessors = [Scaler(255.),
                     Binarizer(args.binarizer_threshold)]
    if args.sobel:
        preprocessors += [Sobel(args.history_length), Binarizer(0, False)]
    if args.predict_next_frame:
        extractor = Regressor(Extractor,
                              discrete_actions=env.action_space.n,
                              input_preprocessor=preprocessors,
                              output_preprocessor=preprocessors,
                              **extractor_params)
    else:
        extractor = Regressor(Extractor,
                              input_preprocessor=preprocessors,
                              output_preprocessor=preprocessors,
                              **extractor_params)

    path =\
        args.load_path + '/' + extractor.model._scope_name
    restorer = tf.train.import_meta_graph(path + '.meta')
    restorer.restore(extractor.model._session, path)
    extractor.model._restore_collection()
else:
    # Approximator
    extractor_params = dict(name='test',
                            load_path=args.load_path,
                            n_actions=env.action_space.n,
                            optimizer={'name': 'adam',
                                       'lr': 1,
                                       'decay': 1},
                            width=84,
                            height=84,
                            history_length=4)
    extractor = Regressor(
        ConvNet,
        input_preprocessor=[Scaler(255.)],
        **extractor_params
    )

buf = Buffer(args.history_length)

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
FEATURES = int(extractor.model.n_features)
ROLLOUT_TIME = 1000
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release


def plot_features(extractor, a, fig):
    if args.dqn:
        features = extractor.predict(np.expand_dims(buf.get(), axis=0),
                                     features=True).reshape(32, 16)
    else:
        if args.predict_next_frame:
            extr_input = [np.expand_dims(buf.get(), axis=0), np.array([[a]])]
        else:
            extr_input = np.expand_dims(buf.get(), axis=0)
        features = extractor.predict(extr_input, features=True)[0].reshape(8,
                                                                           8)
    plt.imshow(features)
    fig.canvas.draw()
    plt.show(block=False)


def rollout(env, fig):
    global human_agent_action, human_wants_restart, human_sets_pause

    human_wants_restart = False
    obser = env.reset()
    frame = Image.fromarray(obser, 'RGB').convert('L').resize((84, 84))
    frame = np.asarray(frame.getdata(), dtype=np.uint8).reshape(frame.size[1],
                                                                frame.size[0])
    for i in xrange(args.history_length):
        buf.add(frame)
    skip = 0
    for t in range(ROLLOUT_TIME):
        buf.add(frame)
        if not skip:
            a = human_agent_action
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)

        plot_features(extractor, a, fig)

        frame = Image.fromarray(obser, 'RGB').convert('L').resize((84, 84))
        frame = np.asarray(frame.getdata(), dtype=np.uint8).reshape(
            frame.size[1], frame.size[0])
        env.render()
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            import time
            time.sleep(.1)


print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

while 1:
    fig = plt.figure()
    rollout(env, fig)
