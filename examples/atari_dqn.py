import numpy as np

from keras.models import Model
from keras.layers import Input, Convolution2D, Flatten, Dense
from keras.optimizers import Adam

from PyPi.algorithms.dqn import DQN
from PyPi.approximators import Regressor
from PyPi.core.core import Core
from PyPi.environments import *
from PyPi.policy import EpsGreedy
from PyPi.utils import logger
from PyPi.utils.parameters import DecayParameter, Parameter


class ConvNet:
    def __init__(self, n_actions):
        # Build network
        self.input = Input(shape=(4, 110, 84))
        self.u = Input(shape=(1,), dtype='int32')

        self.hidden = Convolution2D(32, (8, 8), padding='valid',
                                    activation='relu', strides=(4, 4),
                                    data_format='channels_first')(self.input)

        self.hidden = Convolution2D(64, (4, 4), padding='valid',
                                    activation='relu', strides=(2, 2),
                                    data_format='channels_first')(self.hidden)

        self.hidden = Convolution2D(64, (3, 3), padding='valid',
                                    activation='relu', strides=(1, 1),
                                    data_format='channels_first')(self.hidden)

        self.hidden = Flatten()(self.hidden)
        self.features = Dense(512, activation='relu')(self.hidden)
        self.output = Dense(n_actions, activation='linear')(self.features)
        #self.output = GatherLayer(n_actions)([self.output, self.u])

        # Models
        self.model = Model(outputs=[self.output], inputs=[self.input, self.u])

        # Optimization algorithm
        self.optimizer = Adam()

        # Compile
        self.model.compile(optimizer=self.optimizer, loss='mse',
                           metrics=['mse'])


def experiment():
    np.random.seed()

    # DQN Parameters
    initial_dataset_size = int(5e5)
    target_update_frequency = int(1e5)
    max_dataset_size = int(1e6)
    evaluation_update_frequency = int(5e4)
    max_steps = int(50e6)

    mdp_name = 'BreakoutDeterministic-v3'
    # MDP train
    mdp = Atari(mdp_name, train=True)
    # MDP test
    mdp_test = Atari(mdp_name)

    # Policy
    epsilon = Parameter(value=1)
    pi = EpsGreedy(epsilon=epsilon, observation_space=mdp.observation_space,
                   action_space=mdp.action_space)

    # Approximator
    approximator_params = dict(n_actions=mdp.action_space.n)
    approximator = Regressor(ConvNet, **approximator_params)

    # Agent
    algorithm_params = dict(target_approximator=Regressor(
                                ConvNet, **approximator_params),
                            initial_dataset_size=initial_dataset_size,
                            target_update_frequency=target_update_frequency)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = DQN(approximator, pi, **agent_params)

    # Algorithm
    core = Core(agent, mdp, max_dataset_size=max_dataset_size)
    core_test = Core(agent, mdp_test)

    # DQN
    core.learn(n_iterations=evaluation_update_frequency, how_many=1,
               n_fit_steps=1, iterate_over='samples')
    core_test.evaluate()
    n_steps = evaluation_update_frequency
    agent.policy.set_epsilon(DecayParameter(value=1, decay_exp=0.1))
    for i in xrange(max_steps - evaluation_update_frequency):
        core.learn(n_iterations=evaluation_update_frequency, how_many=1,
                   n_fit_steps=1, iterate_over='samples')
        core_test.evaluate()

        n_steps += evaluation_update_frequency

if __name__ == '__main__':
    logger.Logger(3)
    experiment()
