import numpy as np

from mushroom.environments import Environment
from mushroom.utils import spaces


class ShipSteering(Environment):
    def __init__(self, small=True):
        self.__name__ = 'ShipSteering'

        # MDP spaces
        self.field_size = 150 if small else 1000
        low = np.array([0, 0, -np.pi, -np.pi / 12.])
        high = np.array([self.field_size, self.field_size, np.pi, np.pi / 12.])
        self.omega_max = np.array([np.pi / 12.])
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Box(low=-self.omega_max, high=self.omega_max)

        # MDP parameters
        self.horizon = 5000
        self.gamma = 0.99

        # MDP properties
        self._v = 3.
        self._T = 5.
        self._dt = .2

        self.gate_s = np.empty(2)
        self.gate_e = np.empty(2)
        self.gate_s[0] = 100 if small else 900
        self.gate_s[1] = 120 if small else 920
        self.gate_e[0] = 120 if small else 920
        self.gate_e[1] = 100 if small else 900

        super(ShipSteering, self).__init__()

    def reset(self, state=None):
        if state is None:
            self._state = np.zeros(4)
        else:
            self._state = state

        return self._state

    def step(self, action):
        r = np.maximum(-self.omega_max, np.minimum(self.omega_max, action[0]))
        new_state = np.empty(4)
        new_state[0] = self._state[0] + self._v * np.sin(self._state[2]) *\
            self._dt
        new_state[1] = self._state[1] + self._v * np.cos(self._state[2]) *\
            self._dt
        new_state[2] = self._state[2] + self._state[3] * self._dt
        new_state[3] = self._state[3] + (r - self._state[3]) * self._dt /\
            self._T

        if new_state[0] > self.field_size or new_state[1] > self.field_size\
           or new_state[0] < 0 or new_state[1] < 0:
            reward = -100
            absorbing = True
        elif self._through_gate(self._state[:2], new_state[:2]):
            reward = 0
            absorbing = True
        else:
            reward = -1
            absorbing = False

        self._state = new_state

        return self._state, reward, absorbing, {}

    def _through_gate(self, start, end):
        r = self.gate_e - self.gate_s
        s = end - start
        den = self._cross_2d(vecr=r, vecs=s)

        if den == 0:
            return False

        t = self._cross_2d((start - self.gate_s), s) / den
        u = self._cross_2d((start - self.gate_s), r) / den

        return 1 >= u >= 0 and 1 >= t >= 0

    @staticmethod
    def _cross_2d(vecr, vecs):
        return vecr[0] * vecs[1] - vecr[1] * vecs[0]
