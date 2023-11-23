import numpy as np
from scipy.integrate import odeint

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils import spaces
from mushroom_rl.utils.angles import normalize_angle
from mushroom_rl.utils.viewer import Viewer


class InvertedPendulum(Environment):
    """
    The Inverted Pendulum environment (continuous version) as presented in:
    "Reinforcement Learning In Continuous Time and Space". Doya K.. 2000.
    "Off-Policy Actor-Critic". Degris T. et al.. 2012.
    "Deterministic Policy Gradient Algorithms". Silver D. et al. 2014.

    """
    def __init__(self, random_start=False, m=1., l=1., g=9.8, mu=1e-2,
                 max_u=5., horizon=5000, gamma=.99):
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
            horizon (int, 5000): horizon of the problem;
            gamma (int, .99): discount factor.

        """
        # MDP parameters
        self._m = m
        self._l = l
        self._g = g
        self._mu = mu
        self._random = random_start
        self._max_u = max_u
        self._max_omega = 5 / 2 * np.pi
        high = np.array([np.pi, self._max_omega])

        # MDP properties
        dt = .01
        observation_space = spaces.Box(low=-high, high=high)
        action_space = spaces.Box(low=np.array([-max_u]), high=np.array([max_u]))
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)

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
            self._state[1] = self._bound(self._state[1], -self._max_omega, self._max_omega)

        self._last_u = 0.0
        return self._state, {}

    def step(self, action):
        u = self._bound(action[0], -self._max_u, self._max_u)
        new_state = odeint(self._dynamics, self._state, [0, self.info.dt], args=(u.item(),))

        self._state = np.array(new_state[-1])
        self._state[0] = normalize_angle(self._state[0])
        self._state[1] = self._bound(self._state[1], -self._max_omega, self._max_omega)

        reward = np.cos(self._state[0])

        self._last_u = u.item()

        return self._state, reward, False, {}

    def render(self, record=False):
        start = 1.25 * self._l * np.ones(2)
        end = 1.25 * self._l * np.ones(2)

        end[0] += self._l * np.sin(self._state[0])
        end[1] += self._l * np.cos(self._state[0])

        self._viewer.line(start, end)
        self._viewer.circle(start, self._l / 40)
        self._viewer.circle(end, self._l / 20)
        self._viewer.torque_arrow(start, -self._last_u, self._max_u, self._l / 5)

        frame = self._viewer.get_frame() if record else None

        self._viewer.display(self.info.dt)

        return frame

    def stop(self):
        self._viewer.close()

    def _dynamics(self, state, t, u):
        theta = state[0]
        omega = self._bound(state[1], -self._max_omega, self._max_omega)

        d_theta = omega
        d_omega = (-self._mu * omega + self._m * self._g * self._l * np.sin(theta) + u) / (self._m * self._l**2)

        ds = [d_theta, d_omega]

        return ds
