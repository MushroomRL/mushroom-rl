import numpy as np
from scipy.integrate import odeint

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils import spaces
from mushroom_rl.utils.angles import normalize_angle
from mushroom_rl.utils.viewer import Viewer


class CartPole(Environment):
    """
    The Inverted Pendulum on a Cart environment as presented in:
    "Least-Squares Policy Iteration". Lagoudakis M. G. and Parr R.. 2003.

    """
    def __init__(self, m=2., M=8., l=.5, g=9.8, mu=1e-2, max_u=50., noise_u=10.,
                 horizon=3000, gamma=.95):
        """
        Constructor.

        Args:
            m (float, 2.0): mass of the pendulum;
            M (float, 8.0): mass of the cart;
            l (float, .5): length of the pendulum;
            g (float, 9.8): gravity acceleration constant;
            max_u (float, 50.): maximum allowed input torque;
            noise_u (float, 10.): maximum noise on the action;
            horizon (int, 3000): horizon of the problem;
            gamma (float, .95): discount factor.

        """
        # MDP parameters
        self._m = m
        self._M = M
        self._l = l
        self._g = g
        self._alpha = 1 / (self._m + self._M)
        self._mu = mu
        self._max_u = max_u
        self._noise_u = noise_u
        high = np.array([np.inf, np.inf])

        # MDP properties
        dt = .1
        observation_space = spaces.Box(low=-high, high=high)
        action_space = spaces.Discrete(3)
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)

        # Visualization
        self._viewer = Viewer(2.5 * l, 2.5 * l)
        self._last_u = None
        self._state = None

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            angle = np.random.uniform(-np.pi / 8., np.pi / 8.)

            self._state = np.array([angle, 0.])
        else:
            self._state = state
            self._state[0] = normalize_angle(self._state[0])

        self._last_u = 0
        return self._state, {}

    def step(self, action):
        if action == 0:
            u = -self._max_u
        elif action == 1:
            u = 0.
        else:
            u = self._max_u

        self._last_u = u

        u += np.random.uniform(-self._noise_u, self._noise_u)
        new_state = odeint(self._dynamics, self._state, [0, self.info.dt], (u,))

        self._state = np.array(new_state[-1])
        self._state[0] = normalize_angle(self._state[0])

        if np.abs(self._state[0]) > np.pi * .5:
            reward = -1.
            absorbing = True
        else:
            reward = 0.
            absorbing = False

        return self._state, reward, absorbing, {}

    def render(self, record=False):
        start = 1.25 * self._l * np.ones(2)
        end = 1.25 * self._l * np.ones(2)

        end[0] += self._l * np.sin(self._state[0])
        end[1] += self._l * np.cos(self._state[0])

        self._viewer.line(start, end)
        self._viewer.square(start, 0,  self._l / 10)
        self._viewer.circle(end, self._l / 20)

        direction = -np.sign(self._last_u) * np.array([1, 0])
        value = np.abs(self._last_u)
        self._viewer.force_arrow(start, direction, value, self._max_u, self._l / 5)

        frame = self._viewer.get_frame() if record else None

        self._viewer.display(self.info.dt)

        return frame

    def stop(self):
        self._viewer.close()

    def _dynamics(self, state, t, u):
        theta = state[0]
        omega = state[1]

        d_theta = omega
        d_omega = (self._g * np.sin(theta)
                   - self._alpha * self._m * self._l * .5 * d_theta ** 2 * np.sin(2 * theta) * .5
                   - self._alpha * np.cos(theta) * u) / (2 / 3 * self._l -
                                                         self._alpha * self._m * self._l * .5 * np.cos(theta) ** 2)

        return d_theta, d_omega
