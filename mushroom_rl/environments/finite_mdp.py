import math
import warnings

import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.core.spaces import Discrete
from mushroom_rl.utils.viewer import Viewer


class FiniteMDP(Environment):
    r"""
    Finite Markov Decision Process.

    A Markov Decision Process :math:`\mathcal{M}` is the tuple

    .. math::
        \mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{R}, \mathcal{P}, \iota, \gamma \rangle

    where :math:`\mathcal{S}` is the space of the states of the process, :math:`\mathcal{A}` is the space of the
    actions the agent can take, :math:`\mathcal{R}(s, a, s')` is the reward given for taking the action :math:`a` in
    the state :math:`s` and landing in the state :math:`s'`, :math:`\mathcal{P}(s'|s, a)` is the probability of that
    transition, :math:`\iota` is the initial state distribution, giving the probability of beginning an episode in
    each state, and :math:`\gamma` is the discount factor weighting how much a future reward is worth now.

    Both spaces are finite, :math:`\mathcal{S} = \{0, \dots, |\mathcal{S}| - 1\}` and
    :math:`\mathcal{A} = \{0, \dots, |\mathcal{A}| - 1\}`, so a state and an action are indexes and the process is
    stored in the arrays ``p``, ``r`` and ``iota``:

    .. math::
        p_{s a s'} = \mathcal{P}(s'|s, a), \qquad r_{s a s'} = \mathcal{R}(s, a, s'), \qquad \iota_s = \iota(s)

    where :math:`\sum_{s' \in \mathcal{S}} p_{s a s'} = 1` for every :math:`s \in \mathcal{S}` and
    :math:`a \in \mathcal{A}`.

    A state :math:`\bar{s} \in \mathcal{S}` is absorbing when

    .. math::
        \mathcal{P}(\bar{s}|\bar{s}, a) = 1 \qquad \forall a \in \mathcal{A}

    i.e. it is a sink state. By definition an absorbing state collects no reward,
    :math:`\mathcal{R}(\bar{s}, a, s') = 0`, hence its value is zero and the episode is cut when it is reached. This
    class enforces the definition, zeroing the rows of ``r`` of every absorbing state of ``p``.

    A finite MDP is drawn as a grid of cells: by default the states are laid out on a grid that fills the screen,
    wrapping like text, and the agent's cell is highlighted. A subclass can change the layout by passing
    ``viewer_shape`` (the number of rows and columns of the grid) and the drawing by overriding ``_draw``; the cells,
    their colors and the agent are the only things a finite MDP draws, so the drawing is described by the ``style``
    dictionary rather than by dedicated viewer classes.

    """
    def __init__(self, p, rew, iota=None, gamma=.9, horizon=np.inf, dt=1e-1, viewer_shape=None, **viewer_params):
        """
        Constructor.

        Args:
            p (np.ndarray): transition probability matrix;
            rew (np.ndarray): reward matrix;
            iota (np.ndarray, None): initial state probability distribution;
            gamma (float, .9): discount factor;
            horizon (int, np.inf): the horizon;
            dt (float, 1e-1): the control timestep of the environment;
            viewer_shape (tuple, None): the (n_rows, n_columns) of the grid of cells the states are drawn on. When it
                is not given, the states are laid out on a grid that fills the screen, wrapping like text. When the
                grid is too big to fit the screen, rendering does nothing;
            **viewer_params: parameters forwarded to the viewer, e.g. its size bounds (see ``Viewer``).

        """
        assert p.shape == rew.shape
        assert iota is None or p.shape[0] == iota.size
        assert np.allclose(p.sum(axis=2), 1.), 'The transitions of every state and action must sum to one.'

        # MDP parameters
        self.p = p.copy()
        self.r = rew.copy()
        self.iota = iota if iota is None else iota.copy()

        self._enforce_absorbing()

        # Visualization
        self._style = self._build_style()

        viewer_params.setdefault('min_scale', 40)

        assert viewer_params['min_scale'] >= 1, 'An environment unit must take at least one pixel.'

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
            if self.iota is not None:
                self._state = np.array(
                    [np.random.choice(self.iota.size, p=self.iota)])
            else:
                self._state = np.array([np.random.choice(self.p.shape[0])])
        else:
            self._state = state

        return self._state, {}

    def step(self, action):
        p = self.p[self._state[0], action[0], :]
        next_state = np.array([np.random.choice(p.size, p=p)])
        absorbing = np.all(self.p[next_state[0], :, next_state[0]] == 1.).item()
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

    def _enforce_absorbing(self):
        """
        Make the process obey the definition of an absorbing state. An outcome that is the only one an action can
        lead to is given probability exactly one, so that a state looping on itself is recognised as absorbing
        however the probabilities that built it were rounded, and every absorbing state is stripped of its reward.
        Override it to keep a different convention.

        """
        certain = np.count_nonzero(self.p, axis=2) == 1
        self.p[certain] = self.p[certain] > 0.

        states = np.arange(self.p.shape[0])
        self.r[(self.p[states, :, states] == 1.).all(axis=1)] = 0.

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
