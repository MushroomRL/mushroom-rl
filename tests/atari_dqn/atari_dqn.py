import os

import gym
import numpy as np
import tensorflow as tf

from examples.atari_dqn.convnet import ConvNet
from mushroom.algorithms.value import AveragedDQN, DQN, DoubleDQN
from mushroom.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.parameters import LinearDecayParameter, Parameter
from mushroom.utils.preprocessor import Scaler

# Disable tf cpp warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def experiment(alg):
    gym.logger.setLevel(0)
    np.random.seed(88)
    tf.set_random_seed(88)

    # DQN settings
    initial_replay_size = 500
    max_replay_size = 1000
    train_frequency = 50
    target_update_frequency = 100
    evaluation_frequency = 200
    max_steps = 2000

    # MDP train
    mdp = Atari('BreakoutDeterministic-v4', 84, 84, ends_at_life=True)

    # Policy
    epsilon = LinearDecayParameter(value=1, min_value=.1, n=10)
    epsilon_test = Parameter(value=.05)
    epsilon_random = Parameter(value=1)
    pi = EpsGreedy(epsilon=epsilon_random)

    # Approximator
    input_shape = (84, 84, 4)
    approximator_params = dict(input_shape=input_shape,
                               output_shape=(mdp.info.action_space.n,),
                               n_actions=mdp.info.action_space.n,
                               input_preprocessor=[Scaler(
                                   mdp.info.observation_space.high[0, 0])],
                               optimizer={'name': 'rmsprop',
                                          'lr': .00025,
                                          'decay': .95,
                                          'epsilon': 1e-10}
                               )

    approximator = ConvNet

    # Agent
    algorithm_params = dict(
        batch_size=32,
        initial_replay_size=initial_replay_size,
        n_approximators=2 if alg == 'adqn' else 1,
        max_replay_size=max_replay_size,
        history_length=4,
        train_frequency=train_frequency,
        target_update_frequency=target_update_frequency,
        max_no_op_actions=10,
        no_op_action_value=0
    )

    if alg == 'dqn':
        agent = DQN(approximator, pi, mdp.info,
                    approximator_params=approximator_params, **algorithm_params)
    elif alg == 'ddqn':
        agent = DoubleDQN(approximator, pi, mdp.info,
                          approximator_params=approximator_params,
                          **algorithm_params)
    elif alg == 'adqn':
        agent = AveragedDQN(approximator, pi, mdp.info,
                            approximator_params=approximator_params,
                            **algorithm_params)

    # Algorithm
    core = Core(agent, mdp)

    # DQN

    # fill replay memory with random dataset
    core.learn(n_steps=initial_replay_size,
               n_steps_per_fit=initial_replay_size, quiet=True)

    # evaluate initial policy
    pi.set_epsilon(epsilon_test)
    mdp.set_episode_end(ends_at_life=False)
    for n_epoch in xrange(1, max_steps / evaluation_frequency + 1):
        # learning step
        pi.set_epsilon(epsilon)
        mdp.set_episode_end(ends_at_life=True)
        core.learn(n_steps=evaluation_frequency,
                   n_steps_per_fit=train_frequency, quiet=True)

        # evaluation step
        pi.set_epsilon(epsilon_test)
        mdp.set_episode_end(ends_at_life=False)
    w = agent.approximator.model.get_weights(only_trainable=True)

    return w


if __name__ == '__main__':
    print('Executing atari_dqn test...')

    res = experiment('dqn')
    test_res = np.load('tests/atari_dqn/w.npy')
    tf.reset_default_graph()
    d_res = experiment('ddqn')
    d_test_res = np.load('tests/atari_dqn/dw.npy')
    tf.reset_default_graph()
    a_res = experiment('adqn')
    a_test_res = np.load('tests/atari_dqn/aw.npy')

    for i in xrange(len(res)):
        assert np.array_equal(res[i], test_res[i])
        assert np.array_equal(d_res[i], d_test_res[i])
        assert np.array_equal(a_res[i], a_test_res[i])
