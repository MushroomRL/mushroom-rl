import numpy as np

from mushroom.environments import Environment, MDPInfo
from mushroom.utils import spaces


class GridWorld(Environment):
    def __init__(self, height, width, goal, start=(0, 0)):
        self.__name__ = 'GridWorld'

        # MDP parameters
        self._height = height
        self._width = width
        self._goal = goal
        self._start = start

        # MDP properties
        observation_space = spaces.Discrete([height, width])
        action_space = spaces.Discrete(4)
        horizon = 100
        gamma = .9
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        assert not np.array_equal(self._start, self._goal)

        assert self._goal[0] < self._height and self._goal[1] < self._width,\
            'Goal position not suitable for the grid world dimension.'

        super(GridWorld, self).__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            self._state = np.array(self._start)
        else:
            self._state = state

        return self._state

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

        return self._state, reward, absorbing, {}


class GridWorldVanHasselt(Environment):
    def __init__(self, height=3, width=3, goal=(0, 2), start=(2, 0)):
        self.__name__ = 'GridWorldVanHasselt'

        # MDP parameters
        self._height = height
        self._width = width
        self._goal = goal
        self._start = start

        # MDP properties
        observation_space = spaces.Discrete([height, width])
        action_space = spaces.Discrete(4)
        horizon = np.inf
        gamma = .95
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        assert not np.array_equal(self._start, self._goal)

        assert self._goal[0] < self._height and self._goal[1] < self._width,\
            'Goal position not suitable for the grid world dimension.'

        super(GridWorldVanHasselt, self).__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            self._state = np.array(self._start)
        else:
            self._state = state

        return self._state

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

        return self._state, reward, absorbing, {}


class GridWorldGenerator(Environment):
    def __init__(self, grid_map):
        self.__name__ = 'GridWorldGenerator'

        self._generate(grid_map)

        # MDP properties
        observation_space = spaces.Discrete([self._height, self._width])
        action_space = spaces.Discrete(4)
        horizon = 100
        gamma = .9
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super(GridWorldGenerator, self).__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            self._state = np.array(self._start)
        else:
            self._state = state

        return self._state

    def step(self, action):
        new_state = self._state.copy()
        if action == 0:
            new_state[0] -= 1
        elif action == 1:
            new_state[0] += 1
        elif action == 2:
            new_state[1] -= 1
        elif action == 3:
            new_state[1] += 1

        c = self._grid[new_state[0]][new_state[1]]
        if c == '*':
            reward = -10
            absorbing = True
        elif c in ['.', 'S']:
            reward = 0
            absorbing = False
            self._state = new_state
        elif c == 'G':
            reward = 10
            absorbing = True
            self._state = new_state
        elif c == '#':
            reward = 0
            absorbing = False

        return self._state, reward, absorbing, {}

    def _generate(self, grid_map):
        self._grid = list()
        with open(grid_map, 'r') as f:
            m = f.read()

            assert 'S' in m and 'G' in m

            row = list()
            row_idx = 0
            col_idx = 0
            for c in m:
                if c in ['#', '.', 'S', 'G', '*']:
                    row.append(c)
                    if c == 'S':
                        self._start = (row_idx, col_idx)
                    elif c == 'G':
                        self._goal = (row_idx, col_idx)
                    col_idx += 1
                elif c == '\n':
                    self._grid.append(row)
                    row = list()
                    row_idx += 1
                    col_idx = 0
                else:
                    raise ValueError('Unknown marker.')

        self._height = len(self._grid)
        self._width = 0
        for w in self._grid:
            if len(w) > self._width:
                self._width = len(w)

        assert not np.array_equal(self._start, self._goal)
