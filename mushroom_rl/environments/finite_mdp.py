import math
import warnings

import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.core.spaces import Discrete
from mushroom_rl.utils.viewer import Viewer


class FiniteMDP(Environment):
    """
    Finite Markov Decision Process.

    A finite MDP is drawn as a grid of cells: by default the states are laid out on a grid that fills the screen,
    wrapping like text, and the agent's cell is highlighted. A subclass can change the layout by passing
    ``viewer_shape`` (the number of rows and columns of the grid) and the drawing by overriding ``_draw``; the cells,
    their colors and the agent are the only things a finite MDP draws, so the drawing is described by the ``style``
    dictionary rather than by dedicated viewer classes.

    """
    def __init__(self, p, rew, mu=None, gamma=.9, horizon=np.inf, dt=1e-1, viewer_shape=None, **viewer_params):
        """
        Constructor.

        Args:
            p (np.ndarray): transition probability matrix;
            rew (np.ndarray): reward matrix;
            mu (np.ndarray, None): initial state probability distribution;
            gamma (float, .9): discount factor;
            horizon (int, np.inf): the horizon;
            dt (float, 1e-1): the control timestep of the environment;
            viewer_shape (tuple, None): the (n_rows, n_columns) of the grid of cells the states are drawn on. When it
                is not given, the states are laid out on a grid that fills the screen, wrapping like text. When the
                grid is too big to fit the screen, rendering does nothing;
            **viewer_params: parameters forwarded to the viewer, e.g. its size bounds (see ``Viewer``).

        """
        assert p.shape == rew.shape
        assert mu is None or p.shape[0] == mu.size

        # MDP parameters
        self.p = p
        self.r = rew
        self.mu = mu

        # Visualization
        self._style = self._build_style()

        viewer_params.setdefault('min_scale', 40)

        if viewer_shape is None:
            n_columns = min(p.shape[0], viewer_params.get('max_width', 1920) // viewer_params['min_scale'])
            viewer_shape = (math.ceil(p.shape[0] / n_columns), n_columns)

        self._n_rows, self._n_columns = viewer_shape
        self._viewer = Viewer(self._n_columns, self._n_rows, **viewer_params)

        if not self._viewer.fits:
            warnings.warn(f'{type(self).__name__} is too big to fit the screen and will not be rendered.')

            self._viewer = None

        # MDP properties
        observation_space = Discrete(p.shape[0])
        action_space = Discrete(p.shape[1])
        horizon = horizon
        gamma = gamma
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            if self.mu is not None:
                self._state = np.array(
                    [np.random.choice(self.mu.size, p=self.mu)])
            else:
                self._state = np.array([np.random.choice(self.p.shape[0])])
        else:
            self._state = state

        return self._state, {}

    def step(self, action):
        p = self.p[self._state[0], action[0], :]
        next_state = np.array([np.random.choice(p.size, p=p)])
        absorbing = not np.any(self.p[next_state[0]])
        reward = self.r[self._state[0], action[0], next_state[0]]

        self._state = next_state

        return self._state, reward, absorbing, {}

    def render(self, record=False):
        if self._viewer is None:
            return None

        self._draw()

        frame = self._viewer.get_frame() if record else None

        self._viewer.display(self.info.dt)

        return frame

    def stop(self):
        if self._viewer is not None:
            self._viewer.close()

    def _draw(self):
        """
        Draw the grid of cells and the agent on its cell. Override it to draw a richer representation of the state.

        """
        self._viewer.grid(self._n_rows, self._n_columns, self._style['grid_color'], self._style['line_width'])
        self._viewer.circle(self._cell_center(*self._cell_of(self._state.item())), self._style['agent_radius'],
                            self._style['agent_color'])

    def _cell_center(self, row, column):
        """
        Convert the position of a cell of the grid into the world coordinates of its center. Cells are indexed from the
        top-left corner, while the world is indexed from the bottom-left one.

        Args:
            row (int): row of the cell;
            column (int): column of the cell.

        Returns:
            The coordinates of the center of the cell.

        """
        return np.array([column + .5, self._n_rows - row - .5])

    def _cell_of(self, state):
        """
        Convert a state into the (row, column) of the cell drawing it, reading the grid like text.

        Args:
            state (int): the state of the environment.

        Returns:
            The row and the column of the cell.

        """
        return divmod(state, self._n_columns)

    @classmethod
    def _build_style(cls):
        """
        Build the colors and sizes used to draw the environment. Override it, merging the result of the base class, to
        change the way the environment looks.

        Returns:
            A dictionary describing the drawing.

        """
        return dict(agent_color=(0, 0, 255), grid_color=(255, 255, 255), agent_radius=.4, line_width=1)
