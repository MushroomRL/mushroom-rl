import numpy as np

from mushroom_rl.environments.grid_world import GridWorld


class GridWorldVanHasselt(GridWorld):
    """
    A variant of the grid world as presented in:
    "Double Q-Learning". Hasselt H. V.. 2010.

    Every non-terminating step gives a reward of -12 or +10 with equal probability. Reaching the goal does not end the
    episode: it is the next action, taken in the goal cell, that yields +5 and terminates. The optimal policy therefore
    takes five actions, four to walk to the goal and one to leave it, and the optimal value of the starting state is
    ``5 * gamma ** 4 - sum(gamma ** k for k in range(4))``.

    Leaving the goal is modelled with a terminal state appended to the cells of the grid. It shares its cell with the
    goal, so it is drawn on top of it, but walking on that cell still leads to the goal. The random reward is drawn in
    ``step``, whereas the reward matrix ``r`` stores its expectation, so that dynamic programming on ``p`` and ``r``
    gives the true optimal value function.

    """
    def __init__(self, height=3, width=3, goal=(0, 2), start=(2, 0), gamma=.95, horizon=np.inf, dt=1e-1):
        """
        Constructor.

        Args:
            height (int, 3): height of the grid;
            width (int, 3): width of the grid;
            goal (tuple, (0, 2)): 2D coordinates of the goal cell;
            start (tuple, (2, 0)): 2D coordinates of the starting cell;
            gamma (float, .95): discount factor;
            horizon (int, np.inf): the horizon;
            dt (float, 1e-1): the control timestep of the environment.

        """
        self._step_rewards = np.array([-12., 10.])

        base_map = self._build_grid(height, width, goal, start)
        grid_map = self._build_layers(base_map)

        super().__init__(grid_map, goal_reward=5., gamma=gamma, horizon=horizon, dt=dt)

    def step(self, action):
        state, reward, absorbing, info = super().step(action)

        if not absorbing:
            reward = np.random.choice(self._step_rewards)

        return state, reward, absorbing, info

    def _compute_probabilities(self):
        """
        Compute the transition probability matrix. The goal cell is absorbing for the grid world, so it comes back
        without transitions and every action taken in it is sent to the terminal state.

        Returns:
            The transition probability matrix.

        """
        transition_probabilities = super()._compute_probabilities()

        # the terminal state shares its cell with the goal, so the state map resolves it back to the goal
        goal_state = self._get_state_id(self._cell_list[-1])

        transition_probabilities[goal_state, :, -1] = 1.

        return transition_probabilities

    def _compute_reward(self):
        """
        Compute the reward matrix. Every transition gives the expected step reward, except leaving the goal cell, which
        gives the goal reward.

        Returns:
            The reward matrix.

        """
        n_states = len(self._cell_list)
        reward = np.full((n_states, len(self._directions), n_states), self._step_rewards.mean())
        reward[:, :, -1] = self._goal_reward

        return reward

    @classmethod
    def _build_cell_list(cls, grid_map):
        """
        List the cells of the grid, plus the terminal state reached by acting in the goal cell. The terminal state has
        no cell of its own, so it is given the goal one and drawn on top of it.

        Args:
            grid_map (np.ndarray): (n_layers, height, width) array of symbols describing the map.

        Returns:
            The (n_states, 3) array giving the (layer, row, column) of every state.

        """
        cell_list = super()._build_cell_list(grid_map)
        goal_cell = np.argwhere(grid_map == 'G')[0]

        return np.vstack([cell_list, goal_cell])

    @classmethod
    def _build_state_map(cls, grid_map, cell_list):
        """
        Build the state map of the grid alone, leaving the terminal state out: it shares its cell with the goal, and
        walking on that cell must lead to the goal.

        Args:
            grid_map (np.ndarray): (n_layers, height, width) array of symbols describing the map;
            cell_list (np.ndarray): (n_states, 3) array giving the (layer, row, column) of every state.

        Returns:
            The (n_layers, height, width) array giving the state of every cell.

        """
        return super()._build_state_map(grid_map, cell_list[:-1])
