import os

import numpy as np
import tensorflow as tf

from convnet import ConvNet
from mushroom.algorithms.dqn import DQN
from mushroom.approximators import Regressor
from mushroom.core.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.parameters import LinearDecayParameter, Parameter
from mushroom.utils.preprocessor import Scaler

# Disable tf cpp warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def experiment():
    np.random.seed(88)
    tf.set_random_seed(88)

    # DQN settings
    initial_replay_size = 50
    max_replay_size = 200
    train_frequency = 5
    target_update_frequency = 10
    evaluation_frequency = 50
    max_steps = 200

    # MDP train
    mdp = Atari('BreakoutDeterministic-v4', 84, 84, ends_at_life=True)

    # Policy
    epsilon = LinearDecayParameter(value=1,
                                   min_value=.1,
                                   n=10)
    epsilon_test = Parameter(value=.05)
    epsilon_random = Parameter(value=1)
    pi = EpsGreedy(epsilon=epsilon_random,
                   observation_space=mdp.observation_space,
                   action_space=mdp.action_space)

    # Approximator
    approximator_params_train = dict(n_actions=mdp.action_space.n,
                                     optimizer={'name': 'rmsprop',
                                                'lr': .00025,
                                                'decay': .95},
                                     name='train',
                                     width=84,
                                     height=84,
                                     history_length=4)
    approximator = Regressor(ConvNet,
                             preprocessor=[Scaler(
                                 mdp.observation_space.high)],
                             **approximator_params_train)

    # target approximator
    approximator_params_target = dict(n_actions=mdp.action_space.n,
                                      optimizer={'name': 'rmsprop',
                                                 'lr': .00025,
                                                 'decay': .95},
                                      name='test',
                                      width=84,
                                      height=84,
                                      history_length=4)
    target_approximator = Regressor(
        ConvNet,
        preprocessor=[Scaler(mdp.observation_space.high)],
        **approximator_params_target)

    target_approximator.model.set_weights(approximator.model.get_weights())

    # Agent
    algorithm_params = dict(
        batch_size=32,
        target_approximator=target_approximator,
        initial_replay_size=initial_replay_size,
        max_replay_size=max_replay_size,
        history_length=4,
        train_frequency=train_frequency,
        target_update_frequency=target_update_frequency,
        max_no_op_actions=10,
        no_op_action_value=0
    )
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}

    agent = DQN(approximator, pi, **agent_params)

    # Algorithm
    core = Core(agent, mdp)

    # DQN

    # fill replay memory with random dataset
    core.learn(n_iterations=1, how_many=initial_replay_size,
               n_fit_steps=0, iterate_over='samples', quiet=True)

    # evaluate initial policy
    pi.set_epsilon(epsilon_test)
    mdp.set_episode_end(ends_at_life=False)
    for n_epoch in xrange(1, max_steps / evaluation_frequency + 1):
        # learning step
        pi.set_epsilon(epsilon)
        mdp.set_episode_end(ends_at_life=True)
        core.learn(n_iterations=evaluation_frequency / train_frequency,
                   how_many=train_frequency,
                   n_fit_steps=1,
                   iterate_over='samples',
                   quiet=True)

        # evaluation step
        pi.set_epsilon(epsilon_test)
        mdp.set_episode_end(ends_at_life=False)
    w = approximator.model.get_weights()

    return w

if __name__ == '__main__':
    print('Executing atari_dqn test...')

    res = experiment()
    test_res = np.load('tests/atari_dqn/w.npy')

    for i in xrange(len(res)):
        assert np.array_equal(res[i], test_res[i])
