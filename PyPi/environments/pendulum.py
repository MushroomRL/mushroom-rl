import gym
import numpy as np

from PyPi.utils import spaces


class Pendulum(gym.Env):
    def __init__(self):
        self.__name__ = 'Pendulum-v0'

        # MPD creation
        self.env = gym.make(self.__name__).env

        # MDP spaces
        high = np.array([np.pi, self.env.max_speed])
        self.action_space = spaces.Box(low=-self.env.max_torque,
                                       high=self.env.max_torque, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        # MDP parameters
        self.horizon = 100
        self.gamma = 0.95

        # MDP initialization
        self.env._seed()
        self.reset()

    def reset(self, state=None):
        if state is None:
            self.env._reset()
        else:
            self.env.state = state

        return self.get_state()

    def step(self, action):
        _, reward, absorbing, info = self.env._step(action)

        return self.get_state(), reward, absorbing, info

    def get_state(self):
        return np.array([self.env.state.ravel()])

    def render(self, mode='human', close=False):
        self.env._render(mode=mode, close=close)

    def __str__(self):
        return self.__name__
