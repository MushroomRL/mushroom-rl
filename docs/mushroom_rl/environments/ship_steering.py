import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils import spaces
from mushroom_rl.utils.angles import normalize_angle
from mushroom_rl.utils.viewer import Viewer


class ShipSteering(Environment):
    """
    The Ship Steering environment as presented in:
    "Hierarchical Policy Gradient Algorithms". Ghavamzadeh M. and Mahadevan S.. 2013.

    """
    def __init__(self, small=True, n_steps_action=3):
        """
        Constructor.

        Args:
             small (bool, True): whether to use a small state space or not.
             n_steps_action (int, 3): number of integration intervals for each
                                      step of the env.

        """
        # MDP parameters
        self.field_size = 150 if small else 1000
        low = np.array([0, 0, -np.pi, -np.pi / 12.])
        high = np.array([self.field_size, self.field_size, np.pi, np.pi / 12.])
        self.omega_max = np.array([np.pi / 12.])
        self._v = 3.
        self._T = 5.
        self._gate_s = np.empty(2)
        self._gate_e = np.empty(2)
        self._gate_s[0] = 100 if small else 350
        self._gate_s[1] = 120 if small else 400
        self._gate_e[0] = 120 if small else 450
        self._gate_e[1] = 100 if small else 400
        self._out_reward = -100
        self._success_reward = 0
        self._small = small
        self._state = None
        self.n_steps_action = n_steps_action

        # MDP properties
        dt = .2
        observation_space = spaces.Box(low=low, high=high)
        action_space = spaces.Box(low=-self.omega_max, high=self.omega_max)
        horizon = 5000
        gamma = .99
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)

        # Visualization
        self._viewer = Viewer(self.field_size, self.field_size,
                              background=(66, 131, 237))

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            if self._small:
                self._state = np.zeros(4)
                self._state[2] = np.pi/2
            else:
                low = self.info.observation_space.low
                high = self.info.observation_space.high
                self._state = (high-low)*np.random.rand(4) + low
        else:
            self._state = state

        return self._state, {}

    def step(self, action):

        r = self._bound(action[0], -self.omega_max, self.omega_max)

        new_state = self._state

        for _ in range(self.n_steps_action):
            state = new_state
            new_state = np.empty(4)
            new_state[0] = state[0] + self._v * np.cos(state[2]) * self.info.dt
            new_state[1] = state[1] + self._v * np.sin(state[2]) * self.info.dt
            new_state[2] = normalize_angle(state[2] + state[3] * self.info.dt)
            new_state[3] = state[3] + (r - state[3]) * self.info.dt / self._T

            if new_state[0] > self.field_size or new_state[1] > self.field_size or new_state[0] < 0 or new_state[1] < 0:
                new_state[0] = self._bound(new_state[0], 0, self.field_size)
                new_state[1] = self._bound(new_state[1], 0, self.field_size)

                reward = self._out_reward
                absorbing = True
                break

            elif self._through_gate(state[:2], new_state[:2]):
                reward = self._success_reward
                absorbing = True
                break
            else:
                reward = -1
                absorbing = False

        self._state = new_state

        return self._state, reward, absorbing, {}

    def render(self, record=False):
        self._viewer.line(self._gate_s, self._gate_e,
                          width=3)

        boat = [
            [-4, -4],
            [-4, 4],
            [4, 4],
            [8, 0.0],
            [4, -4]
        ]
        self._viewer.polygon(self._state[:2], self._state[2], boat,
                             color=(32, 193, 54))

        frame = self._viewer.get_frame() if record else None

        self._viewer.display(self.info.dt)

        return frame

    def stop(self):
        self._viewer.close()

    def _through_gate(self, start, end):
        r = self._gate_e - self._gate_s
        s = end - start
        den = self._cross_2d(vecr=r, vecs=s)

        if den == 0:
            return False

        t = self._cross_2d((start - self._gate_s), s) / den
        u = self._cross_2d((start - self._gate_s), r) / den

        return 1 >= u >= 0 and 1 >= t >= 0

    @staticmethod
    def _cross_2d(vecr, vecs):
        return vecr[0] * vecs[1] - vecr[1] * vecs[0]
