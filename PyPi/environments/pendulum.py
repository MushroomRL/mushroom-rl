import numpy as np
from gym.envs.classic_control import PendulumEnv

from PyPi.utils import spaces


class Pendulum(PendulumEnv):
    def __init__(self):
        super(Pendulum, self).__init__()

        # MDP spaces
        high = np.array([np.pi, self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque,
                                       high=self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        # MDP parameters
        self.horizon = 100
        self.gamma = 0.95

        # MDP initialization
        self.reset()

    def reset(self, state=None):
        if state is None:
            self._reset()
        else:
            self.state = state

        return self.get_state()

    def step(self, action):
        _, reward, absorbing, info = self._step(action)

        return self.get_state(), reward, absorbing, info

    def get_state(self):
        return np.array([self.state.ravel()])
