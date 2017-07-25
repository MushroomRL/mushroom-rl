import gym
from gym.utils import seeding

import numpy as np


class Environment(gym.Env):
    def __init__(self):
        # MDP initialization
        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_state(self):
        state = np.array(self._state)
        if state.ndim == 0:
            return np.array([state])
        elif state.ndim > 0:
            return state
        else:
            raise ValueError('Wrong state dimension.')

    def get_info(self):
        return {'observation_space': self.observation_space,
                'action_space': self.action_space,
                'gamma': self.gamma,
                'horizon': self.horizon}

    def __str__(self):
        return self.__name__