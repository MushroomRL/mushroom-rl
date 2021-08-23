import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils import spaces
from mushroom_rl.utils.viewer import Viewer


class AbstractGridWorld(Environment):
    """
    Abstract class to build a grid world.

    """
    def __init__(self, mdp_info, height, width, start, goal):
        """
        Constructor.

        Args:
            height (int): height of the grid;
            width (int): width of the grid;
            start (tuple): x-y coordinates of the goal;
            goal (tuple): x-y coordinates of the goal.

        """
        assert not np.array_equal(start, goal)

        assert goal[0] < height and goal[1] < width,\
            'Goal position not suitable for the grid world dimension.'

        self._state = None
        self._height = height
        self._width = width
        self._start = start
        self._goal = goal

        # Visualization
        self._viewer = Viewer(self._width, self._height, 500,
                              self._height * 500 // self._width)

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            state = self.convert_to_int(self._start, self._width)

        self._state = state

        return self._state

    def step(self, action):
        state = self.convert_to_grid(self._state, self._width)

        new_state, reward, absorbing, info = self._step(state, action)
        self._state = self.convert_to_int(new_state, self._width)

        return self._state, reward, absorbing, info

    def render(self):
        for row in range(1, self._height):
            for col in range(1, self._width):
                self._viewer.line(np.array([col, 0]),
                                  np.array([col, self._height]))
                self._viewer.line(np.array([0, row]),
                                  np.array([self._width, row]))

        goal_center = np.array([.5 + self._goal[1],
                                self._height - (.5 + self._goal[0])])
        self._viewer.square(goal_center, 0, 1, (0, 255, 0))

        start_grid = self.convert_to_grid(self._start, self._width)
        start_center = np.array([.5 + start_grid[1],
                                 self._height - (.5 + start_grid[0])])
        self._viewer.square(start_center, 0, 1, (255, 0, 0))

        state_grid = self.convert_to_grid(self._state, self._width)
        state_center = np.array([.5 + state_grid[1],
                                 self._height - (.5 + state_grid[0])])
        self._viewer.circle(state_center, .4, (0, 0, 255))

        self._viewer.display(.1)

    def _step(self, state, action):
        raise NotImplementedError('AbstractGridWorld is an abstract class.')

    def _grid_step(self, state, action):
        action = action[0]
        if action == 0:
            if state[0] > 0:
                state[0] -= 1
        elif action == 1:
            if state[0] + 1 < self._height:
                state[0] += 1
        elif action == 2:
            if state[1] > 0:
                state[1] -= 1
        elif action == 3:
            if state[1] + 1 < self._width:
                state[1] += 1

    @staticmethod
    def convert_to_grid(state, width):
        return np.array([state[0] // width, state[0] % width])

    @staticmethod
    def convert_to_int(state, width):
        return np.array([state[0] * width + state[1]])


class GridWorld(AbstractGridWorld):
    """
    Standard grid world.

    """
    def __init__(self, height, width, goal, start=(0, 0)):
        # MDP properties
        observation_space = spaces.Discrete(height * width)
        action_space = spaces.Discrete(4)
        horizon = 100
        gamma = .9
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info, height, width, start, goal)

    def _step(self, state, action):
        self._grid_step(state, action)

        if np.array_equal(state, self._goal):
            reward = 10
            absorbing = True
        else:
            reward = 0
            absorbing = False

        return state, reward, absorbing, {}


class GridWorldVanHasselt(AbstractGridWorld):
    """
    A variant of the grid world as presented in:
    "Double Q-Learning". Hasselt H. V.. 2010.

    """
    def __init__(self, height=3, width=3, goal=(0, 2), start=(2, 0)):
        # MDP properties
        observation_space = spaces.Discrete(height * width)
        action_space = spaces.Discrete(4)
        horizon = np.inf
        gamma = .95
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info, height, width, start, goal)

    def _step(self, state, action):
        if np.array_equal(state, self._goal):
            reward = 5
            absorbing = True
        else:
            self._grid_step(state, action)

            reward = np.random.choice([-12, 10])
            absorbing = False

        return state, reward, absorbing, {}
