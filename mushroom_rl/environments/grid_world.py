import numpy as np

from mushroom_rl.environments.finite_mdp import FiniteMDP


class GridWorld(FiniteMDP):
    """
    Grid world environment.

    The world is described by a map of symbols. There are five types of cells: 'S' is a starting cell, 'G' is a goal
    cell, '.' is a normal cell, '*' is a hole and '#' is a wall. Reaching a goal or a hole ends the episode, while
    walking into a wall leaves the agent where it is. The initial state is drawn uniformly among the starting cells.

    The map is a stack of layers, so that the agent position can be combined with a configuration of the world.
    The layer is part of the state: the same position may represent a different environment state.

    Every step of the construction of the Markov Decision Process is a method of this class, so that a different grid
    world can be obtained by subclassing and overriding the interesting ones: ``_build_legend`` to add new symbols,
    ``_build_layers`` to stack more configurations of the world, and ``_compute_probabilities``, ``_compute_reward``
    and ``_compute_iota`` to change the dynamics, the reward and the initial state distribution.

    """
    def __init__(self, grid_map, prob=1., goal_reward=1., hole_reward=-1., gamma=.9, horizon=100, dt=1e-1,
                 **viewer_params):
        """
        Constructor.

        Args:
            grid_map (np.ndarray): (n_layers, height, width) array of symbols describing the map;
            prob (float, 1.): probability of success of an action. When an action fails, the agent does not move;
            goal_reward (float, 1.): reward obtained when reaching a goal cell;
            hole_reward (float, -1.): reward obtained when falling into a hole;
            gamma (float, .9): discount factor;
            horizon (int, 100): the horizon;
            dt (float, 1e-1): the control timestep of the environment;
            **viewer_params: parameters forwarded to the viewer, e.g. its size bounds (see ``Viewer``).

        """
        assert np.any(grid_map == 'G'), 'The map must contain at least one goal cell.'

        self._grid_map = grid_map
        self._prob = prob
        self._goal_reward = goal_reward
        self._hole_reward = hole_reward

        self._legend = self._build_legend()
        self._cell_list = self._build_cell_list(grid_map)
        self._directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        self._state_map = self._build_state_map(grid_map, self._cell_list)

        transition_probabilities = self._compute_probabilities()
        reward = self._compute_reward()
        initial_distribution = self._compute_iota()

        _, height, width = grid_map.shape

        super().__init__(transition_probabilities, reward, initial_distribution, gamma, horizon, dt,
                         viewer_shape=(height, width), **viewer_params)

    @classmethod
    def generate(cls, height=3, width=3, goal=None, start=(0, 0), **kwargs):
        """
        Build the standard version of the grid world, an empty square grid with the goal in the corner opposite to
        the starting cell.

        Args:
            height (int, 3): height of the grid;
            width (int, 3): width of the grid;
            goal (tuple, None): 2D coordinates of the goal cell, None to place it in the corner opposite to the
                starting cell;
            start (tuple, (0, 0)): 2D coordinates of the starting cell;
            **kwargs: the parameters of the constructor.

        Returns:
            The standard grid world.

        """
        if goal is None:
            goal = (height - 1, width - 1)

        return cls.from_size(height, width, goal, start, **kwargs)

    @classmethod
    def from_file(cls, path, **kwargs):
        """
        Build a grid world reading the map from a text file. Every line of the file is a row of the grid, and the grid
        is expected to be rectangular.

        Args:
            path (str): the path of the file containing the map;
            **kwargs: the parameters of the constructor.

        Returns:
            The grid world described by the file.

        """
        base_map = cls._read_map(path)
        grid_map = cls._build_layers(base_map)

        return cls(grid_map=grid_map, **kwargs)

    @classmethod
    def from_size(cls, height, width, goal, start=(0, 0), **kwargs):
        """
        Build an empty rectangular grid world with a single starting cell and a single goal cell.

        Args:
            height (int): height of the grid;
            width (int): width of the grid;
            goal (tuple): 2D coordinates of the goal cell;
            start (tuple, (0, 0)): 2D coordinates of the starting cell;
            **kwargs: the parameters of the constructor.

        Returns:
            The grid world of the given size.

        """
        base_map = cls._build_grid(height, width, goal, start)
        grid_map = cls._build_layers(base_map)

        return cls(grid_map=grid_map, **kwargs)

    @property
    def grid_map(self):
        """
        Returns:
            The map of the grid world.

        """
        return self._grid_map

    @property
    def cell_list(self):
        """
        Returns:
            The (layer, row, column) of every state of the grid world.

        """
        return self._cell_list

    def _compute_probabilities(self):
        """
        Compute the transition probability matrix of the grid world. The cells marked as terminal in the map are
        absorbing, so they loop on themselves whatever the action.

        Returns:
            The transition probability matrix.

        """
        n_states = len(self._cell_list)
        prob = np.zeros((n_states, len(self._directions), n_states))

        for state, cell in enumerate(self._cell_list):
            if self._marked_as_terminal(cell):
                prob[state, :, state] = 1.
            else:
                for action, direction in enumerate(self._directions):
                    prob[state, action] = self._compute_action_probabilities(cell, direction)

        return prob

    def _compute_action_probabilities(self, cell, direction):
        """
        Compute the probability of reaching every state when taking an action in a cell. Override it to change what
        an action can do, e.g. to let it slip in a direction other than the intended one.

        Args:
            cell (np.ndarray): the (layer, row, column) of the current cell;
            direction (np.ndarray): the 2D displacement of the action.

        Returns:
            The probability of every state being the next one.

        """
        probabilities = np.zeros(len(self._cell_list))
        state = self._get_state_id(cell)
        next_cell = self._next_cell(cell, direction)

        if next_cell is None:
            probabilities[state] = 1.
        else:
            probabilities[state] += 1. - self._prob
            probabilities[self._get_state_id(next_cell)] += self._prob

        return probabilities

    def _compute_reward(self):
        """
        Compute the reward matrix of the grid world.

        Returns:
            The reward matrix.

        """
        n_states = len(self._cell_list)
        reward = np.zeros((n_states, len(self._directions), n_states))

        self._reward_on_arrival(reward, 'G', self._goal_reward)
        self._reward_on_arrival(reward, '*', self._hole_reward)

        return reward

    def _compute_iota(self):
        """
        Compute the initial state distribution of the grid world, uniform among the starting cells. The episode always
        begins in the first layer of the map, which is therefore the initial configuration of the world.

        Returns:
            The initial state distribution.

        """
        initial_distribution = np.zeros(len(self._cell_list))

        for state, cell in enumerate(self._cell_list):
            if cell[0] == 0 and self._grid_map[tuple(cell)] == 'S':
                initial_distribution[state] = 1.

        assert initial_distribution.sum() > 0, 'The map must contain at least one starting cell.'

        return initial_distribution / initial_distribution.sum()

    def _draw(self):
        """
        Draw the map of the layer the agent is in, coloring every cell according to its symbol, on top of which the
        grid of cells and the agent are drawn.

        """
        layer = self._cell_list[self._state.item(), 0]

        for cell_row in range(self._n_rows):
            for cell_column in range(self._n_columns):
                color = self._style['colors'].get(self._grid_map[layer, cell_row, cell_column])

                if color is not None:
                    self._viewer.square(self._cell_center(cell_row, cell_column), 0, 1, color)

        super()._draw()

    def _cell_of(self, state):
        """
        Convert a state into the (row, column) of the cell drawing it, reading it from the map rather than from the
        position of the state in the grid, because the cells that are not a state, e.g. the walls, are skipped.

        Args:
            state (int): the state of the environment.

        Returns:
            The row and the column of the cell.

        """
        return tuple(self._cell_list[state, 1:])

    def _next_cell(self, cell, direction):
        """
        Compute the cell reached by the agent moving in the given direction.

        Args:
            cell (np.ndarray): the (layer, row, column) of the current cell;
            direction (np.ndarray): the 2D displacement of the move.

        Returns:
            The (layer, row, column) of the cell reached by the agent, or None if the move is blocked.

        """
        next_cell = cell.copy()
        next_cell[1:] += direction

        _, height, width = self._grid_map.shape
        in_bounds = 0 <= next_cell[1] < height and 0 <= next_cell[2] < width

        if in_bounds and self._legend[self._grid_map[tuple(next_cell)]]['walkable']:
            return next_cell

        return None

    def _get_state_id(self, cell):
        """
        Find the state the agent is in when it stands on the given cell.

        Args:
            cell (np.ndarray): the (layer, row, column) of the cell.

        Returns:
            The state of the cell, or None if the cell is not a state of the environment.

        """
        state = self._state_map[tuple(cell)]

        return state if state >= 0 else None

    def _marked_as_terminal(self, cell):
        """
        Check whether the given cell is marked as terminal in the map, i.e. reaching it ends the episode. This reads
        the map legend, and is what makes the cell absorbing when the dynamics are built.

        Args:
            cell (np.ndarray): the (layer, row, column) of the cell.

        Returns:
            True if the cell is marked as terminal, False otherwise.

        """
        return self._legend[self._grid_map[tuple(cell)]]['absorbing']

    def _reward_on_arrival(self, reward, symbol, value):
        """
        Give a reward to every transition reaching a cell with the given symbol.

        Args:
            reward (np.ndarray): the reward matrix to fill;
            symbol (str): the symbol of the rewarding cells;
            value (float): the reward to give.

        """
        for state, cell in enumerate(self._cell_list):
            if self._grid_map[tuple(cell)] == symbol:
                reward[:, :, state] = value

    @classmethod
    def _read_map(cls, path):
        """
        Read a text file into an array of symbols, checking that the map is rectangular and that every symbol is
        known. Blank lines are allowed at the end of the file only, so that a blank line left in the middle of a map
        is reported rather than closing the gap it leaves.

        Args:
            path (str): the path of the file containing the map.

        Returns:
            The (height, width) array of symbols described by the file.

        """
        with open(path, 'r') as grid_file:
            text = grid_file.read().rstrip('\n')

        assert text, 'The map is empty.'

        rows = [list(row) for row in text.split('\n')]
        width = max(len(row) for row in rows)

        for number, row in enumerate(rows, 1):
            if len(row) != width:
                raise ValueError(f'Row {number} of the map is {len(row)} cells long instead of {width}.')

        grid_map = np.array(rows)

        legend = cls._build_legend()
        for symbol in np.unique(grid_map):
            if symbol not in legend:
                raise ValueError(f'Unknown symbol "{symbol}" in the map.')

        return grid_map

    @classmethod
    def _build_grid(cls, height, width, goal, start=(0, 0)):
        """
        Build the map of an empty rectangular grid with a single starting cell and a single goal cell.

        Args:
            height (int): height of the grid;
            width (int): width of the grid;
            goal (tuple): 2D coordinates of the goal cell;
            start (tuple, (0, 0)): 2D coordinates of the starting cell.

        Returns:
            The (height, width) array of symbols describing the grid.

        """
        assert not np.array_equal(start, goal), 'The starting cell and the goal cell must be different.'
        assert goal[0] < height and goal[1] < width, 'Goal position not suitable for the grid world dimension.'
        assert start[0] < height and start[1] < width, 'Start position not suitable for the grid world dimension.'

        grid_map = np.full((height, width), '.')
        grid_map[start[0], start[1]] = 'S'
        grid_map[goal[0], goal[1]] = 'G'

        return grid_map

    @classmethod
    def _build_layers(cls, base_map):
        """
        Stack the layers of the map, one for every configuration the world can be in. A plain grid world has a single
        configuration, so the map of a grid world is its only layer.

        Args:
            base_map (np.ndarray): (height, width) array of symbols describing the grid.

        Returns:
            The (n_layers, height, width) array of symbols describing the map.

        """
        return base_map[np.newaxis]

    @classmethod
    def _build_cell_list(cls, grid_map):
        """
        List every cell of the map the agent can occupy, i.e. every cell that is not a wall. The position of a cell in
        this list is the state of the Markov Decision Process.

        Args:
            grid_map (np.ndarray): (n_layers, height, width) array of symbols describing the map.

        Returns:
            The (n_states, 3) array giving the (layer, row, column) of every state.

        """
        legend = cls._build_legend()
        cell_list = [cell for cell in np.ndindex(grid_map.shape) if legend[grid_map[cell]]['walkable']]

        return np.array(cell_list)

    @classmethod
    def _build_state_map(cls, grid_map, cell_list):
        """
        Build the inverse of the cell list, giving the state of every cell of the map. The cells that are not a state,
        e.g. the walls, are marked with -1.

        Args:
            grid_map (np.ndarray): (n_layers, height, width) array of symbols describing the map;
            cell_list (np.ndarray): (n_states, 3) array giving the (layer, row, column) of every state.

        Returns:
            The (n_layers, height, width) array giving the state of every cell.

        """
        layers, rows, columns = cell_list.T

        state_map = np.full(grid_map.shape, -1)
        state_map[layers, rows, columns] = np.arange(len(cell_list))

        return state_map

    @classmethod
    def _build_legend(cls):
        """
        Build the meaning of every symbol of the map. A cell is walkable if the agent can stand on it, and absorbing if
        reaching it ends the episode.

        Returns:
            A dictionary mapping each symbol to its meaning.

        """
        return {
            '#': dict(walkable=False, absorbing=False),
            '.': dict(walkable=True, absorbing=False),
            'S': dict(walkable=True, absorbing=False),
            'G': dict(walkable=True, absorbing=True),
            '*': dict(walkable=True, absorbing=True)
        }

    @classmethod
    def _build_style(cls):
        style = super()._build_style()
        style['colors'] = {
            '#': (105, 105, 105),   # wall
            'S': (245, 222, 179),   # starting cell
            'G': (0, 255, 0),       # goal
            '*': (255, 0, 0)        # hole
        }

        return style
