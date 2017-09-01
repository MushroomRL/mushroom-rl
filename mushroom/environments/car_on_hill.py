import numpy as np
from scipy.integrate import odeint

from mushroom.environments import Environment
from mushroom.utils import spaces


class CarOnHill(Environment):
    """
    The Car On Hill environment as presented in:
    "Tree-Based Batch Mode Reinforcement Learning, D. Ernst et. al."
    """
    def __init__(self):
        self.__name__ = 'CarOnHill'

        # MDP spaces
        self.max_pos = 1.
        self.max_velocity = 3.
        high = np.array([self.max_pos, self.max_velocity])
        self.observation_space = spaces.Box(low=-high, high=high)
        self.action_space = spaces.Discrete(2)

        # MDP parameters
        self.horizon = 100
        self.gamma = 0.95

        # MDP properties
        self._g = 9.81
        self._m = 1
        self._dt = .1
        self._discrete_actions = [-4., 4.]

        super(CarOnHill, self).__init__()

    def reset(self, state=None):
        if state is None:
            self._state = np.array([-0.5, 0])
        else:
            self._state = state

        return self._state

    def step(self, action):
        action = self._discrete_actions[action[0]]
        sa = np.append(self._state, action)
        new_state = odeint(self._dpds, sa, [0, self._dt])

        self._state = new_state[-1, :-1]

        if self._state[0] < -self.max_pos or \
                np.abs(self._state[1]) > self.max_velocity:
            reward = -1
            absorbing = True
        elif self._state[0] > self.max_pos and \
                np.abs(self._state[1]) <= self.max_velocity:
            reward = 1
            absorbing = True
        else:
            reward = 0
            absorbing = False

        return self._state, reward, absorbing, {}

    def _dpds(self, state_action, t):
        position = state_action[0]
        velocity = state_action[1]
        u = state_action[-1]

        if position < 0.:
            diff_hill = 2 * position + 1
            diff_2_hill = 2
        else:
            diff_hill = 1 / ((1 + 5 * position ** 2) ** 1.5)
            diff_2_hill = (-15 * position) / ((1 + 5 * position ** 2) ** 2.5)

        dp = velocity
        ds = (u - self._g * self._m * diff_hill - velocity ** 2 * self._m *
              diff_hill * diff_2_hill) / (self._m * (1 + diff_hill ** 2))

        return dp, ds, 0.
