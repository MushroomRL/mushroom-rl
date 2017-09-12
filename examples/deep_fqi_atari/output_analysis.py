import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from examples.deep_fqi_atari.extractor import Extractor
from mushroom.approximators.action_regressor import ActionRegressor
from mushroom.environments import Atari
from mushroom.utils.preprocessor import Binarizer, Scaler

load_path = '/home/shirokuma/Projects/RL/mushroom/logs/'
binarizer_threshold = .1

# MDP
mdp = Atari('BreakoutDeterministic-v4', 84, 84)
mdp.reset()

extractor_params = dict(folder_name=None,
                        n_actions=mdp.action_space.n,
                        optimizer={'name': 'adam',
                                   'lr': 1,
                                   'decay': 1},
                        width=84,
                        height=84,
                        history_length=4)
extractor = ActionRegressor(Extractor,
                            discrete_actions=mdp.action_space.n,
                            input_preprocessor=[
                                Scaler(255.),
                                Binarizer(binarizer_threshold)
                            ],
                            output_preprocessor=[
                                Scaler(255.),
                                Binarizer(binarizer_threshold)
                            ],
                            **extractor_params)

for i, e in enumerate(extractor.models):
    restorer = tf.train.import_meta_graph(
        load_path + e.model._scope_name + '/' + e.model._scope_name + '.meta')
    restorer.restore(e.model._session, load_path +
                     e.model._scope_name + '/' + e.model._scope_name)
    e.model._restore_collection()

# Predictions
n_samples = 100
state = np.ones((n_samples, 84, 84, 4))
action = np.ones(n_samples)
next_state = np.ones((n_samples, 84, 84, 4))

for i in xrange(state.shape[0]):
    for j in xrange(4):
        state[i, ..., j], _, _, _ = mdp.step(np.random.randint(4))
    next_state[i, ..., :3] = state[i, ..., 1:].copy()
    a = np.random.randint(4)
    next_state[i, ..., 3], _, _, _ = mdp.step(a)
    action[i] = a

action_samples = list()
for i in xrange(mdp.action_space.n):
    idxs = np.argwhere(action == i).ravel()
    action_samples.append(idxs[0])
    print('Model %d loss: %f' % (i, extractor.models[i].model.get_loss(
        (next_state[idxs] / 255. >= binarizer_threshold).astype(np.float),
        extractor.models[i].predict(state[idxs], reconstruction=True)))
    )

for a in action_samples:
    plt.figure()
    prediction = extractor.models[action[a].astype(np.int)].model.predict(
        np.expand_dims(state[a], axis=0),
        reconstruction=True
    )
    for i in xrange(4):
        plt.subplot(2, 4, i + 1)
        plt.imshow(state[a, ..., i])
        plt.subplot(2, 4, i + 5)
        plt.imshow(prediction[0, ..., i])

plt.show()
