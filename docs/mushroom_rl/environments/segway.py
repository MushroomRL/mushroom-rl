import numpy as np
from scipy.integrate import odeint

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils import spaces
from mushroom_rl.utils.angles import normalize_angle
from mushroom_rl.utils.viewer import Viewer


class Segway(Environment):
    """
    The Segway environment (continuous version) as presented in:
    "Deep Learning for Actor-Critic Reinforcement Learning". Xueli Jia. 2015.

    """
    def __init__(self, random_start=False):
        """
        Constructor.

        Args:
            random_start (bool, False): whether to start from a random position
                or from the horizontal one.

        """
        # MDP parameters
        gamma = 0.97

        self._Mr = 0.3 * 2
        self._Mp = 2.55
        self._Ip = 2.6e-2
        self._Ir = 4.54e-4 * 2
        self._l = 13.8e-2
        self._r = 5.5e-2
        self._g = 9.81
        self._max_u = 5

        self._random = random_start

        high = np.array([-np.pi / 2, 15, 75])

        # MDP properties
        dt = 1e-2
        observation_space = spaces.Box(low=-high, high=high)
        action_space = spaces.Box(low=np.array([-self._max_u]),
                                  high=np.array([self._max_u]))
        horizon = 300
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)

        # Visualization
        self._viewer = Viewer(5 * self._l, 5 * self._l)
        self._last_x = 0

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            if self._random:
                angle = np.random.uniform(-np.pi / 2, np.pi / 2)
            else:
                angle = -np.pi/8

            self._state = np.array([angle, 0., 0.])
        else:
            self._state = state
            self._state[0] = normalize_angle(self._state[0])

        self._last_x = 0

        return self._state, {}

    def step(self, action):
        u = self._bound(action[0], -self._max_u, self._max_u)
        new_state = odeint(self._dynamics, self._state, [0, self.info.dt], (u,))

        self._state = np.array(new_state[-1])
        self._state[0] = normalize_angle(self._state[0])

        if abs(self._state[0]) > np.pi / 2:
            absorbing = True
            reward = -10000
        else:
            absorbing = False
            Q = np.diag([3.0, 0.1, 0.1])

            x = self._state

            J = x.dot(Q).dot(x)

            reward = -J

        return self._state, reward, absorbing, {}

    def _dynamics(self, state, t, u):

        alpha = state[0]
        d_alpha = state[1]

        h1 = (self._Mr + self._Mp) * (self._r ** 2) + self._Ir
        h2 = self._Mp * self._r * self._l * np.cos(alpha)
        h3 = self._l ** 2 * self._Mp + self._Ip

        omegaP = d_alpha

        dOmegaP = -(h2 * self._l * self._Mp * self._r * np.sin( alpha) * omegaP**2
                    - self._g * h1 * self._l * self._Mp * np.sin(alpha) + (h2 + h1) * u) / (h1 * h3 - h2**2)
        dOmegaR = (h3 * self._l * self._Mp * self._r * np.sin(alpha) * omegaP**2
                   - self._g * h2 * self._l * self._Mp * np.sin(alpha) + (h3 + h2) * u) / (h1 * h3 - h2**2)

        dx = list()
        dx.append(omegaP)
        dx.append(dOmegaP)
        dx.append(dOmegaR)

        return dx

    def render(self, record=False):
        start = 2.5 * self._l * np.ones(2)
        end = 2.5 * self._l * np.ones(2)

        dx = -self._state[2] * self._r * self.info.dt

        self._last_x += dx

        if self._last_x > 2.5 * self._l or self._last_x < -2.5 * self._l:
            self._last_x = (2.5 * self._l + self._last_x) % (5 * self._l) - 2.5 * self._l

        start[0] += self._last_x
        end[0] += -2 * self._l * np.sin(self._state[0]) + self._last_x
        end[1] += 2 * self._l * np.cos(self._state[0])

        if (start[0] > 5 * self._l and end[0] > 5 * self._l)  or (start[0] < 0 and end[0] < 0):
            start[0] = start[0] % 5 * self._l
            end[0] = end[0] % 5 * self._l

        self._viewer.line(start, end)
        self._viewer.circle(start, self._r)

        frame = self._viewer.get_frame() if record else None

        self._viewer.display(self.info.dt)

        return frame

    def stop(self):
        self._viewer.close()



