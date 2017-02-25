import gym
import numpy as np

from PyPi.utils import spaces


class GridWorld(gym.Env):
    def __init__(self, height, width, goal):
        # MDP spaces
        self.observation_space = spaces.MultiDiscrete([[0, height - 1],
                                                       [0, width - 1]])
        self.action_space = spaces.Discrete(4)

        # MDP parameters
        self.horizon = 100
        self.gamma = .9

        # MDP properties
        self._height = height
        self._width = width
        self._goal = goal

        # MDP initialization
        self.seed()
        self.reset()

    def reset(self, state=None):
        if state is None:
            self._state = np.array([0, 0])
        else:
            self._state = state

        return self.get_state()

    def step(self, action):
        if action == 0:
            if self._state[0] - 1 >= 0:
                self._state[0] -= 1
        elif action == 1:
            if self._state[0] + 1 < self._height:
                self._state[0] += 1
        elif action == 2:
            if self._state[1] - 1 >= 0:
                self._state[1] -= 1
        elif action == 3:
            if self._state[1] + 1 < self._width:
                self._state[1] += 1

        if np.array_equal(self._state, self._goal):
            reward = 10
            absorbing = True
        else:
            reward = 0
            absorbing = False

        return self.get_state(), reward, absorbing, {}

    def get_state(self):
        return np.array([self._state])
