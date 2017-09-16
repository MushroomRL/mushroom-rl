#!/usr/bin/env python
from __future__ import print_function

import argparse
import datetime
import sys, gym
import thread

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from PIL import Image

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
args = parser.parse_args()

env = gym.make(args.game)

# Feature extractor
folder_name = './' + datetime.datetime.now().strftime(
    '%Y-%m-%d_%H-%M-%S')
extractor_params = dict(folder_name=None,
                        n_actions=env.action_space.n,
                        optimizer={'name': 'adam',
                                   'lr': 1,
                                   'decay': 1},
                        width=84,
                        height=84,
                        history_length=4)
extractor = Regressor(Extractor,
                      discrete_actions=env.action_space.n,
                      input_preprocessor=[
                          Scaler(255.),
                          Binarizer(args.binarizer_threshold)],
                      output_preprocessor=[
                          Scaler(255.),
                          Binarizer(args.binarizer_threshold)],
                      **extractor_params)

path =\
    args.load_path + '/' + extractor.model._scope_name
restorer = tf.train.import_meta_graph(path + '.meta')
restorer.restore(extractor.model._session, path)
extractor.model._restore_collection()
buf = Buffer(4)

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
    sa = [np.expand_dims(buf.get(), axis=0), np.array([[a]])]
    features = extractor.predict(sa)[0].reshape(32, 16)
    plt.imshow(features)
    fig.canvas.draw()
    plt.show(block=False)

def rollout(env, fig):
    global human_agent_action, human_wants_restart, human_sets_pause

    human_wants_restart = False
    obser = env.reset()
    frame = Image.fromarray(obser, 'RGB').convert('L').resize((84, 84))
    frame = np.asarray(frame.getdata(), dtype=np.uint8).reshape(frame.size[1], frame.size[0])
    for i in xrange(4):
        buf.add(frame)
    skip = 0
    for t in range(ROLLOUT_TIME):
        buf.add(frame)
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)

        plot_features(extractor, a, fig)

        frame = Image.fromarray(obser, 'RGB').convert('L').resize((84, 84))
        frame = np.asarray(frame.getdata(), dtype=np.uint8).reshape(frame.size[1], frame.size[0])
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
