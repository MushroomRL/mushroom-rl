import numpy as np
from scipy.integrate import odeint

from mushroom.environments import Environment, MDPInfo
from mushroom.utils import spaces


class InvertedPendulum(Environment):
    """
    The Inverted Pendulum environment as presented in:
    "Least-Squares Policy Iteration". Lagoudakis M. G. and Parr R.. 2003.

    """
    def __init__(self):
        self.__name__ = 'InvertedPendulum'

        # MDP parameters
        self.max_degree = np.inf
        self.max_angular_velocity = np.inf
        high = np.array([self.max_degree, self.max_angular_velocity])
        self._g = 9.8
        self._m = 2.
        self._M = 8.
        self._l = .5
        self._alpha = 1. / (self._m + self._M)
        self._dt = .1
        self._discrete_actions = [-50., 0., 50.]

        # MDP properties
        observation_space = spaces.Box(low=-high, high=high)
        action_space = spaces.Discrete(3)
        horizon = 3000
        gamma = .95
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super(InvertedPendulum, self).__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            self._state = np.array([0., 0.])
        else:
            self._state = state

        self._state[0] = self._range_pi(self._state[0])

        return self._state

    def step(self, action):
        action = self._discrete_actions[action[0]]
        action += np.random.uniform(-10., 10.)
        sa = np.append(self._state, action)
        new_state = odeint(self._dpds, sa, [0, self._dt])

        self._state = new_state[-1, :-1]
        self._state[0] = self._range_pi(self._state[0])

        if np.abs(self._state[0]) > np.pi / 2.:
            reward = -1
            absorbing = True
        else:
            reward = 0
            absorbing = False

        return self._state, reward, absorbing, {}

    def _dpds(self, state_action, t):
        angle = state_action[0]
        velocity = state_action[1]
        u = state_action[-1]

        dp = velocity
        ds = (
            self._g * np.sin(angle) - self._alpha * self._m * self._l * dp**2 *
            np.sin(2 * angle) * .5 - self._alpha * np.cos(angle) * u) / (
            4 / 3. * self._l - self._alpha * self._m * self._l * np.cos(
                angle)**2)

        return dp, ds, 0.

    @staticmethod
    def _range_pi(angle):
        pi_2 = np.pi * 2
        angle = angle - pi_2 * np.floor(angle / pi_2)

        return angle if angle <= np.pi else angle - pi_2
