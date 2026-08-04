import math

import numpy as np

from mushroom_rl.environments.finite_mdp import FiniteMDP


class SimpleChain(FiniteMDP):
    """
    Simple Chain environment.

    The states are arranged in a line and the agent can either move forward, towards the last state, or backward,
    towards the first one. Each action has a certain probability of success and, when it fails, the agent does not
    move. Trying to move past either end of the chain leaves the agent where it is. A reward is given every time the
    agent reaches a goal state, which does not end the episode.

    The chain is drawn as a single row of cells; a chain too long to fit the width of the screen wraps onto several
    rows, reading like text.

    """
    def __init__(self, n_states=5, goal_states=(2,), prob=.8, goal_reward=1., iota=None, gamma=.9, horizon=100,
                 dt=1e-1, **viewer_params):
        """
        Constructor.

        Args:
            n_states (int, 5): number of states of the chain;
            goal_states (tuple, (2,)): the states giving a reward;
            prob (float, .8): probability of success of an action;
            goal_reward (float, 1.): reward obtained when reaching a goal state;
            iota (np.ndarray, None): initial state probability distribution;
            gamma (float, .9): discount factor;
            horizon (int, 100): the horizon;
            dt (float, 1e-1): the control timestep of the environment;
            **viewer_params: parameters forwarded to the viewer, e.g. its size bounds (see ``Viewer``).

        """
        assert iota is None or len(iota) == n_states

        self._n_states = n_states
        self._goal_states = goal_states
        self._prob = prob
        self._goal_reward = goal_reward

        transition_probabilities = self._compute_probabilities()
        reward = self._compute_reward()

        super().__init__(transition_probabilities, reward, iota, gamma, horizon, dt, **viewer_params)

    def _compute_probabilities(self):
        """
        Compute the transition probability matrix of the chain.

        Returns:
            The transition probability matrix.

        """
        transition_probabilities = np.zeros((self._n_states, 2, self._n_states))

        for state in range(self._n_states):
            forward = min(state + 1, self._n_states - 1)
            backward = max(state - 1, 0)

            transition_probabilities[state, 0, state] += 1. - self._prob
            transition_probabilities[state, 0, forward] += self._prob

            transition_probabilities[state, 1, state] += 1. - self._prob
            transition_probabilities[state, 1, backward] += self._prob

        return transition_probabilities

    def _compute_reward(self):
        """
        Compute the reward matrix of the chain. The reward is given when entering a goal state, so an agent standing
        on a goal state and failing to move gets nothing.

        Returns:
            The reward matrix.

        """
        reward = np.zeros((self._n_states, 2, self._n_states))

        for goal_state in self._goal_states:
            reward[:, :, goal_state] = self._goal_reward
            reward[goal_state, :, goal_state] = 0.

        return reward

    def _draw(self):
        """
        Draw the chain, painting the goal states on top of the default grid of cells and the agent.

        """
        for goal_state in self._goal_states:
            self._viewer.square(self._cell_center(*self._cell_of(goal_state)), 0, 1, self._style['goal_color'])

        super()._draw()

    @classmethod
    def _build_style(cls):
        style = super()._build_style()
        style['goal_color'] = (0, 255, 0)

        return style

    @staticmethod
    def _build_viewer_shape(n_states, max_width, max_height, min_scale):
        """
        Lay the chain out on a single row, wrapping onto as many rows as it takes when it is too long to fit the
        width of the screen, so that it is read like a line of text.

        """
        n_columns = min(n_states, max(1, max_width // min_scale))

        return math.ceil(n_states / n_columns), n_columns
