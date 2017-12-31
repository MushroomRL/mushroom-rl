import numpy as np

from mushroom.environments import Environment, MDPInfo
from mushroom.utils import spaces


class Taxi(Environment):
    """
    The Multi-Passenger Taxi domain environment as presented in:
    "Bayesian Q-Learning". Dearden R. et al.. 1998.

    A .txt file has to be used to specify the shape of the grid world and the
    cells. There are five types of cells: 'S' is the position where the agent
    is; 'D' is the destination state; '.' is a normal cell; 'F' is a passenger,
    when the agent steps on a hole, it receives a negative reward and the
    episode ends; '#' is a wall, when the agent is supposed to step on a wall,
    it actually remains in its current state.

    """
    def __init__(self, grid_map):
        """
        Constructor.

        Args:
            grid_map (str): the path of the .txt file containing the grid
                structure.

        """
        self.__name__ = 'Taxi'

        self._grid, height, width, start, destination, passengers =\
            self._generate(grid_map)

        # MDP parameters
        self._slip_probability = .1
        self._state = None
        self._height = height
        self._width = width
        self._start = start
        self._destination = destination
        self._passengers = passengers
        self._collected_passengers = list()

        assert not np.array_equal(start, destination)
        assert len(passengers) > 0
        assert destination[0] < height and destination[1] < width,\
            'Goal position not suitable for the taxi grid dimension.'

        # MDP properties
        observation_space = spaces.Discrete(height * width)
        action_space = spaces.Discrete(4)
        horizon = np.inf
        gamma = .99
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super(Taxi, self).__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            state = self.convert_to_int(self._start, self._width)

        self._state = state
        self._collected_passengers = list()

        return self._state

    def step(self, action):
        state = self.convert_to_grid(self._state, self._width)

        new_state = np.array(state)
        if action == 0:
            if np.random.rand() < self._slip_probability:
                if np.random.rand() < .5:
                    new_state[1] -= 1
                else:
                    new_state[1] += 1
            else:
                new_state[0] -= 1
        elif action == 1:
            if np.random.rand() < self._slip_probability:
                if np.random.rand() < .5:
                    new_state[1] -= 1
                else:
                    new_state[1] += 1
            else:
                new_state[0] += 1
        elif action == 2:
            if np.random.rand() < self._slip_probability:
                if np.random.rand() < .5:
                    new_state[0] -= 1
                else:
                    new_state[0] += 1
            else:
                new_state[1] -= 1
        elif action == 3:
            if np.random.rand() < self._slip_probability:
                if np.random.rand() < .5:
                    new_state[0] -= 1
                else:
                    new_state[0] += 1
            else:
                new_state[1] += 1

        if not 0 <= new_state[
                    0] < self._height or not 0 <= new_state[1] < self._width:
            reward = 0
            absorbing = False
            new_state = np.array(state)
        else:
            c = self._grid[new_state[0]][new_state[1]]
            if c == 'F':
                if new_state.tolist() not in self._collected_passengers:
                    self._collected_passengers.append(new_state.tolist())
                reward = 0
                absorbing = False
            elif c in ['.', 'S']:
                reward = 0
                absorbing = False
            elif c == 'D':
                if len(self._collected_passengers) == 0:
                    reward = 0
                elif len(self._collected_passengers) == 1:
                    reward = 1
                elif len(self._collected_passengers) == 2:
                    reward = 3
                else:
                    reward = 15
                absorbing = True
            elif c == '#':
                reward = 0
                absorbing = False
                new_state = np.array(state)

        self._state = self.convert_to_int(new_state, self._width)

        return self._state, reward, absorbing, {}

    @staticmethod
    def convert_to_grid(state, width):
        return np.array([state[0] / width, state[0] % width])

    @staticmethod
    def convert_to_int(state, width):
        return np.array([state[0] * width + state[1]])

    @staticmethod
    def _generate(grid_map):
        grid = list()
        passengers = list()
        with open(grid_map, 'r') as f:
            m = f.read()

            assert 'S' in m and 'D' in m and 'F' in m

            row = list()
            row_idx = 0
            col_idx = 0
            for c in m:
                if c in ['#', '.', 'S', 'D', 'F']:
                    row.append(c)
                    if c == 'S':
                        start = (row_idx, col_idx)
                    elif c == 'D':
                        destination = (row_idx, col_idx)
                    elif c == 'F':
                        passengers.append((row_idx, col_idx))
                    col_idx += 1
                elif c == '\n':
                    grid.append(row)
                    row = list()
                    row_idx += 1
                    col_idx = 0
                else:
                    raise ValueError('Unknown marker.')

        height = len(grid)
        width = 0
        for w in grid:
            if len(w) > width:
                width = len(w)

        return grid, height, width, start, destination, passengers
