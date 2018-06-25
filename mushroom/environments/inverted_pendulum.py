import numpy as np
from scipy.integrate import odeint

from mushroom.environments import Environment, MDPInfo
from mushroom.utils import spaces
from mushroom.utils.angles import normalize_angle
from mushroom.utils.viewer import Viewer


class InvertedPendulum(Environment):
    """
    The Inverted Pendulum environment (continuous version) as presented in:
    "Reinforcement Learning In Continuous Time and Space". Doya K.. 2000.
    "Off-Policy Actor-Critic". Degris T. et al.. 2012.
    "Deterministic Policy Gradient Algorithms". Silver D. et al. 2014.

    """
    def __init__(self, random_start=False, m=1.0, l=1.0, g=9.8, mu=1e-2,
                 max_u=5.0, horizon=5000):
        """
        Constructor.

        Args:
            random_start (bool, False): whether to start from a random position
                or from the horizontal one;
            m (float, 1.0): mass of the pendulum;
            l (float, 1.0): length of the pendulum;
            g (float, 9.8): gravity acceleration constant;
            mu (float, 1e-2): friction constant of the pendulum;
            max_u (float, 5.0): maximum allowed input torque;
            horizon (int, 5000): horizon of the problem.

        """
        # MDP parameters
        self._m = m
        self._l = l
        self._g = g
        self._mu = mu
        self._random = random_start
        self._dt = 0.01
        self._max_u = max_u
        self._max_omega = 5 / 2 * np.pi
        high = np.array([np.pi, self._max_omega])

        # MDP properties
        observation_space = spaces.Box(low=-high, high=high)
        action_space = spaces.Box(low=np.array([-max_u]),
                                  high=np.array([max_u]))
        gamma = .99
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        # Visualization
        self._viewer = Viewer(2.5 * l, 2.5 * l)
        self._last_u = None

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            if self._random:
                angle = np.random.uniform(-np.pi, np.pi)
            else:
                angle = np.pi / 2

            self._state = np.array([angle, 0.])
        else:
            self._state = state
            self._state[0] = normalize_angle(self._state[0])
            self._state[1] = self._bound(self._state[1], -self._max_omega,
                                         self._max_omega)

        return self._state

    def step(self, action):
        u = self._bound(action[0], -self._max_u, self._max_u)
        new_state = odeint(self._dynamics, self._state, [0, self._dt],
                           (u,))

        self._state = np.array(new_state[-1])
        self._state[0] = normalize_angle(self._state[0])
        self._state[1] = self._bound(self._state[1], -self._max_omega,
                                     self._max_omega)

        reward = np.cos(self._state[0])

        self._last_u = u

        return self._state, reward, False, {}

    def render(self, mode='human'):
        start = 1.25 * self._l * np.ones(2)
        end = 1.25 * self._l * np.ones(2)

        end[0] += self._l * np.sin(self._state[0])
        end[1] += self._l * np.cos(self._state[0])

        self._viewer.line(start, end)
        self._viewer.circle(start, self._l / 40)
        self._viewer.circle(end, self._l / 20)
        self._viewer.torque_arrow(start, -self._last_u, self._max_u,
                                  self._l / 5)

        self._viewer.display(self._dt)

    def stop(self):
        self._viewer.close()

    def _dynamics(self, state, t, u):
        theta = state[0]
        omega = self._bound(state[1], -self._max_omega, self._max_omega)

        d_theta = omega
        d_omega = (-self._mu * omega + self._m * self._g * self._l * np.sin(
            theta) + u) / (self._m * self._l**2)

        return d_theta, d_omega


class InvertedPendulumDiscrete(Environment):
    """
    The Inverted Pendulum environment as presented in:
    "Least-Squares Policy Iteration". Lagoudakis M. G. and Parr R.. 2003.

    """
    def __init__(self, m=2., M=8., l=.5, g=9.8, mu=1e-2, max_u=50., noise_u=10.,
                 horizon=3000):
        """
        Constructor.

        Args:
            m (float, 2.0): mass of the pendulum;
            M (float, 8.0): mass of the cart;
            l (float, .5): length of the pendulum;
            g (float, 9.8): gravity acceleration constant;
            mu (float, 1e-2): friction constant of the pendulum;
            max_u (float, 50.): maximum allowed input torque;
            noise_u (float, 10.): maximum noise on the action;
            horizon (int, 3000): horizon of the problem.

        """
        # MDP parameters
        self._m = m
        self._M = M
        self._l = l
        self._g = g
        self._alpha = 1 / (self._m + self._M)
        self._mu = mu
        self._dt = .1
        self._max_u = max_u
        self._noise_u = noise_u
        high = np.array([np.inf, np.inf])

        # MDP properties
        observation_space = spaces.Box(low=-high, high=high)
        action_space = spaces.Discrete(3)
        gamma = .95
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        # Visualization
        self._viewer = Viewer(2.5 * l, 2.5 * l)
        self._last_u = None

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            angle = np.random.uniform(-np.pi / 8., np.pi / 8.)

            self._state = np.array([angle, 0.])
        else:
            self._state = state
            self._state[0] = normalize_angle(self._state[0])

        return self._state

    def step(self, action):
        if action == 0:
            u = -self._max_u
        elif action == 1:
            u = 0.
        else:
            u = self._max_u
        u += np.random.uniform(-self._noise_u, self._noise_u)
        new_state = odeint(self._dynamics, self._state, [0, self._dt],
                           (u,))

        self._state = np.array(new_state[-1])
        self._state[0] = normalize_angle(self._state[0])

        if np.abs(self._state[0]) > np.pi * .5:
            reward = -1.
            absorbing = True
        else:
            reward = 0.
            absorbing = False

        self._last_u = u

        return self._state, reward, absorbing, {}

    def render(self, mode='human'):
        start = 1.25 * self._l * np.ones(2)
        end = 1.25 * self._l * np.ones(2)

        end[0] += self._l * np.sin(self._state[0])
        end[1] += self._l * np.cos(self._state[0])

        self._viewer.line(start, end)
        self._viewer.circle(start, self._l / 40)
        self._viewer.circle(end, self._l / 20)
        self._viewer.torque_arrow(start, -self._last_u, self._max_u,
                                  self._l / 5)

        self._viewer.display(self._dt)

    def stop(self):
        self._viewer.close()

    def _dynamics(self, state, t, u):
        theta = state[0]
        omega = state[1]

        d_theta = omega
        d_omega = (self._g * np.sin(theta) - self._alpha * self._m * self._l *
                   d_theta ** 2 * np.sin(2 * theta) * .5 - self._alpha * np.cos(
                    theta) * u) / (4 / 3 * self._l - self._alpha * self._m *
                                   self._l * np.cos(theta) ** 2)

        return d_theta, d_omega
