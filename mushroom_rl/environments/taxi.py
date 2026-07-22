import numpy as np

from mushroom_rl.environments.grid_world import GridWorld


class Taxi(GridWorld):
    """
    The Taxi environment, a grid world where the agent has to reach the goal after picking up as many passengers as
    possible. On top of the symbols of the grid world, 'P' is a passenger, which is picked up by walking on it. The
    reward is zero everywhere, except when reaching the goal, where it depends on the number of passengers collected.
    Each action has a certain probability of success and, when it fails, the agent moves in a direction perpendicular
    to the intended one.

    The passengers already picked up are part of the state: the map has one layer for every subset of collected
    passengers, and picking up a passenger moves the agent to the layer where that passenger is missing.

    This problem is inspired from:
    "Bayesian Q-Learning". Dearden R. et al. 1998.

    """
    def __init__(self, grid_map, prob=.9, goal_rewards=(0, 1, 3, 15), gamma=.99, horizon=5000, dt=1e-1):
        """
        Constructor.

        Args:
            grid_map (np.ndarray): (n_layers, height, width) array of symbols describing the map;
            prob (float, .9): probability of success of an action;
            goal_rewards (tuple, (0, 1, 3, 15)): reward obtained when reaching the goal, indexed by the number of
                collected passengers;
            gamma (float, .99): discount factor;
            horizon (int, 5000): the horizon;
            dt (float, 1e-1): the control timestep of the environment.

        """
        self._passenger_positions = np.argwhere(grid_map[0] == 'P')
        self._goal_rewards = goal_rewards

        assert len(goal_rewards) == len(self._passenger_positions) + 1, \
            'A reward must be given for every possible number of collected passengers.'

        super().__init__(grid_map, prob=prob, gamma=gamma, horizon=horizon, dt=dt)

    @classmethod
    def from_size(cls, height, width, goal, start=(0, 0), passengers=(), **kwargs):
        """
        Build an empty rectangular taxi problem with a single starting cell, a single goal cell and the given
        passengers waiting to be picked up.

        Args:
            height (int): height of the grid;
            width (int): width of the grid;
            goal (tuple): 2D coordinates of the goal cell;
            start (tuple, (0, 0)): 2D coordinates of the starting cell;
            passengers (tuple, ()): 2D coordinates of every passenger;
            **kwargs: the parameters of the constructor.

        Returns:
            The taxi problem of the given size.

        """
        base_map = cls._build_grid(height, width, goal, start, passengers)
        grid_map = cls._build_layers(base_map)

        return cls(grid_map, **kwargs)

    def _compute_probabilities(self):
        n_states = len(self._cell_list)
        transition_probabilities = np.zeros((n_states, len(self._directions), n_states))

        for state, cell in enumerate(self._cell_list):
            if not self._marked_as_terminal(cell):
                for action, direction in enumerate(self._directions):
                    self._add_move(transition_probabilities, state, action, cell, direction, self._prob)

                    for slip in self._perpendicular_directions(direction):
                        self._add_move(transition_probabilities, state, action, cell, slip, (1. - self._prob) * .5)

        return transition_probabilities

    def _compute_reward(self):
        n_states = len(self._cell_list)
        reward = np.zeros((n_states, len(self._directions), n_states))

        for state, cell in enumerate(self._cell_list):
            if self._grid_map[tuple(cell)] == 'G':
                n_collected_passengers = int(cell[0]).bit_count()

                reward[:, :, state] = self._goal_rewards[n_collected_passengers]

        return reward

    def _next_cell(self, cell, direction):
        next_cell = super()._next_cell(cell, direction)

        if next_cell is not None and self._grid_map[tuple(next_cell)] == 'P':
            next_cell[0] |= 1 << self._passenger_index(next_cell[1:])

        return next_cell

    def _add_move(self, transition_probabilities, state, action, cell, direction, prob):
        """
        Add to the transition probability matrix the outcome of moving in the given direction.

        Args:
            transition_probabilities (np.ndarray): the transition probability matrix to fill;
            state (int): the state the taxi moves from;
            action (int): the action the taxi takes;
            cell (np.ndarray): the (layer, row, column) of the current cell;
            direction (np.ndarray): the 2D displacement of the move;
            prob (float): the probability of this outcome.

        """
        next_cell = self._next_cell(cell, direction)
        next_state = state if next_cell is None else self._get_state_id(next_cell)

        transition_probabilities[state, action, next_state] += prob

    def _passenger_index(self, position):
        """
        Find which passenger waits at the given position.

        Args:
            position (np.ndarray): the (row, column) of the passenger.

        Returns:
            The index of the passenger.

        """
        return np.argwhere((self._passenger_positions == position).all(axis=1)).item()

    @classmethod
    def _build_grid(cls, height, width, goal, start=(0, 0), passengers=()):
        """
        Build the map of an empty rectangular grid with a single starting cell, a single goal cell and the given
        passengers waiting to be picked up.

        Args:
            height (int): height of the grid;
            width (int): width of the grid;
            goal (tuple): 2D coordinates of the goal cell;
            start (tuple, (0, 0)): 2D coordinates of the starting cell;
            passengers (tuple, ()): 2D coordinates of every passenger.

        Returns:
            The (height, width) array of symbols describing the grid.

        """
        grid_map = super()._build_grid(height, width, goal, start)

        for passenger in passengers:
            assert passenger[0] < height and passenger[1] < width, \
                'Passenger position not suitable for the grid world dimension.'
            assert grid_map[passenger[0], passenger[1]] == '.', 'A passenger must wait on an empty cell.'

            grid_map[passenger[0], passenger[1]] = 'P'

        return grid_map

    @classmethod
    def _build_layers(cls, base_map):
        """
        Stack one layer of the map for every subset of passengers already picked up. The passengers on board are
        removed from their layer, so that the map the taxi sees only holds the passengers still waiting.

        Args:
            base_map (np.ndarray): (height, width) array of symbols describing the grid.

        Returns:
            The (n_layers, height, width) array of symbols describing the map.

        """
        passenger_positions = np.argwhere(base_map == 'P')
        n_layers = 2 ** len(passenger_positions)

        grid_map = np.repeat(base_map[np.newaxis], n_layers, axis=0)

        for layer in range(n_layers):
            for passenger, position in enumerate(passenger_positions):
                if layer & (1 << passenger):
                    grid_map[layer, position[0], position[1]] = '.'

        return grid_map

    @classmethod
    def _build_cell_list(cls, grid_map):
        """
        List every cell the taxi can occupy. On top of the walls, the cells of the passengers still waiting are left
        out, because walking on one of them immediately picks the passenger up, moving the taxi to another layer.

        Args:
            grid_map (np.ndarray): (n_layers, height, width) array of symbols describing the map.

        Returns:
            The (n_states, 3) array giving the (layer, row, column) of every state.

        """
        legend = cls._build_legend()
        cell_list = [cell for cell in np.ndindex(grid_map.shape)
                     if legend[grid_map[cell]]['walkable'] and grid_map[cell] != 'P']

        return np.array(cell_list)

    @classmethod
    def _build_legend(cls):
        legend = super()._build_legend()
        legend['P'] = dict(walkable=True, absorbing=False)

        return legend

    @classmethod
    def _build_style(cls):
        style = super()._build_style()
        style['colors']['P'] = (255, 255, 0)    # passenger waiting to be picked up

        return style

    @staticmethod
    def _perpendicular_directions(direction):
        """
        Compute the two directions perpendicular to the given one, i.e. where the taxi ends up when an action fails.

        Args:
            direction (np.ndarray): the 2D displacement of the intended move.

        Returns:
            The two perpendicular displacements.

        """
        perpendicular = np.abs(direction)[::-1]

        return np.array([perpendicular, -perpendicular])
