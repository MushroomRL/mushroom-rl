import gym
import numpy as np

from PyPi.environments import Environment
from PyPi.utils import spaces


class Pendulum(Environment):
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

        super(Pendulum, self).__init__()

    def reset(self, state=None):
        if state is None:
            self.env.reset()
        else:
            self.env.state = state

        return self.get_state()

    def step(self, action):
        _, reward, absorbing, info = self.env.step(action)

        return self.get_state(), reward, absorbing, info

    def render(self, mode='human', close=False):
        self.env.render(mode=mode, close=close)
