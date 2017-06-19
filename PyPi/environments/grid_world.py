import numpy as np

from PyPi.environments import Environment
from PyPi.utils import spaces


class GridWorld(Environment):
    def __init__(self, height, width, goal, start=(0, 0)):
        self.__name__ = 'GridWorld'

        # MDP spaces
        self.observation_space = spaces.MultiDiscrete(
            (spaces.Discrete(height), spaces.Discrete(width)))
        self.action_space = spaces.Discrete(4)

        # MDP parameters
        self.horizon = 100
        self.gamma = .9

        # MDP properties
        self._height = height
        self._width = width
        self._goal = goal
        self._start = start

        assert not np.array_equal(self._start, self._goal)

        assert self._goal[0] < self._height and self._goal[1] < self._width,\
            'Goal position not suitable for the grid world dimension.'

        super(GridWorld, self).__init__()

    def reset(self, state=None):
        if state is None:
            self._state = np.array(self._start)
        else:
            self._state = state

        return self.get_state()

    def step(self, action):
        if action == 0:
            if self._state[0] > 0:
                self._state[0] -= 1
        elif action == 1:
            if self._state[0] + 1 < self._height:
                self._state[0] += 1
        elif action == 2:
            if self._state[1] > 0:
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


class GridWorldVanHasselt(Environment):
    def __init__(self, height=3, width=3, goal=(0, 2), start=(2, 0)):
        self.__name__ = 'GridWorldVanHasselt'

        # MDP spaces
        self.observation_space = spaces.MultiDiscrete(
            (spaces.Discrete(height), spaces.Discrete(width)))
        self.action_space = spaces.Discrete(4)

        # MDP parameters
        self.horizon = np.inf
        self.gamma = .95

        # MDP properties
        self._height = height
        self._width = width
        self._goal = goal
        self._start = start

        assert not np.array_equal(self._start, self._goal)

        assert self._goal[0] < self._height and self._goal[1] < self._width,\
            'Goal position not suitable for the grid world dimension.'

        super(GridWorldVanHasselt, self).__init__()

    def reset(self, state=None):
        if state is None:
            self._state = np.array(self._start)
        else:
            self._state = state

        return self.get_state()

    def step(self, action):
        if np.array_equal(self._state, self._goal):
            reward = 5
            absorbing = True
        else:
            if action == 0:
                if self._state[0] > 0:
                    self._state[0] -= 1
            elif action == 1:
                if self._state[0] + 1 < self._height:
                    self._state[0] += 1
            elif action == 2:
                if self._state[1] > 0:
                    self._state[1] -= 1
            elif action == 3:
                if self._state[1] + 1 < self._width:
                    self._state[1] += 1

            reward = np.random.choice([-12, 10])
            absorbing = False

        return self.get_state(), reward, absorbing, {}


class GridWorldVerticalWall(Environment):
    def __init__(self, height, width, wall, door, goal, start=(0, 0)):
        self.__name__ = 'GridWorld'

        # MDP spaces
        self.observation_space = spaces.MultiDiscrete(
            (spaces.Discrete(height), spaces.Discrete(width)))
        self.action_space = spaces.Discrete(4)

        # MDP parameters
        self.horizon = 100
        self.gamma = .9

        # MDP properties
        self._height = height
        self._width = width
        self._goal = goal
        self._start = start
        self._wall = wall
        self._door = door

        assert not np.array_equal(self._start, self._goal)

        super(GridWorldVerticalWall, self).__init__()

    def reset(self, state=None):
        if state is None:
            self._state = np.array(self._start)
        else:
            self._state = state

        return self.get_state()

    def step(self, action):
        reward = 0
        absorbing = False
        if action == 0:
            if self._state[0] > 0:
                if self._check_door_step():
                    self._state[0] -= 1
                else:
                    reward = -10
                    absorbing = True
        elif action == 1:
            if self._state[0] + 1 < self._height:
                if self._check_door_step():
                    self._state[0] += 1
                else:
                    reward = -10
                    absorbing = True
        elif action == 2:
            if self._state[1] > 0:
                if self._check_left_step():
                    self._state[1] -= 1
                else:
                    reward = -10
                    absorbing = True
        elif action == 3:
            if self._state[1] + 1 < self._width:
                if self._check_right_step():
                    self._state[1] += 1
                else:
                    reward = -10
                    absorbing = True

        if np.array_equal(self._state, self._goal):
            reward = 10
            absorbing = True

        return self.get_state(), reward, absorbing, {}

    def _check_left_step(self):
        if self._state[1] - 1 == self._wall:
            return self._state[0] == self._door

    def _check_right_step(self):
        if self._state[1] + 1 == self._wall:
            return self._state[0] == self._door

    def _check_door_step(self):
        return self._state[1] != self._wall
