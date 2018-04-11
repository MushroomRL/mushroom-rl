import numpy as np
from scipy.integrate import odeint

from mushroom.environments import Environment, MDPInfo
from mushroom.utils import spaces
from mushroom.utils.angles_utils import normalize_angle


class InvertedPendulum(Environment):
    """
    The Inverted Pendulum environment (continuous version) as presented in:
    "Reinforcement Learning In Continuous Time and Space". Doya K.. 2000.
    "Off-Policy Actor-Critic". Degris T. et al.. 2012.
    "Deterministic Policy Gradient Algorithms". Silver D. et al. 2014.

    """
    def __init__(self, random_start=False, m=1.0, l=1.0, g=9.8, mu=1e-2,
                 max_u=2.0):
        """
        Constructor.

        Args:
            random_start: whether to start from a random position or from the
                          horizontal one
            m (float, 1.0): Mass of the pendulum
            l (float, 1.0): Length of the pendulum
            g (float, 9.8): gravity acceleration constant
            mu (float, 1e-2): friction constant of the pendulum
            max_u (float, 2.0): maximum allowed input torque

        """
        # MDP parameters
        self._g = g
        self._m = m
        self._l = l
        self._mu = mu
        self._random = random_start
        self._dt = 0.02
        self._max_u = max_u
        self._max_omega = 78.54
        high = np.array([np.pi, self._max_omega])

        # MDP properties
        observation_space = spaces.Box(low=-high, high=high)
        action_space = spaces.Box(low=np.array([-max_u]),
                                  high=np.array([max_u]))
        horizon = 5000
        gamma = .99
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super(InvertedPendulum, self).__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            if self._random:
                angle = np.random.uniform(-np.pi, np.pi)
            else:
                angle = np.pi/2

            self._state = np.array([angle, 0.])
        else:
            self._state = state
            self._state[0] = normalize_angle(self._state[0])
            self._state[1] = np.maximum(-self._max_omega,
                                        np.minimum(self._state[1],
                                                   self._max_omega))

        return self._state

    def step(self, action):

        u = np.maximum(-self._max_u, np.minimum(self._max_u, action[0]))
        new_state = odeint(self._dynamics, self._state, [0, self._dt],
                           (u,))

        self._state = np.array(new_state[-1])
        self._state[0] = normalize_angle(self._state[0])
        self._state[1] = np.maximum(-self._max_omega,
                                    np.minimum(self._state[1],
                                               self._max_omega))

        reward = np.cos(self._state[0])

        return self._state, reward, False, {}

    def _dynamics(self, state, t, u):
        theta = state[0]
        omega = np.maximum(-self._max_omega,
                           np.minimum(state[1], self._max_omega))

        d_theta = omega
        d_omega = (-self._mu*omega + self._m*self._g*self._l*np.sin(theta) + u)\
                  / (self._m*self._l**2)

        return d_theta, d_omega

