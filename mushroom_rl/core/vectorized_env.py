import numpy as np

from .environment import Environment


class VectorizedEnvironment(Environment):
    """
    Class to create a Mushroom environment using the PyBullet simulator.

    """
    def __init__(self, mdp_info, n_envs):
        self._n_envs = n_envs
        super().__init__(mdp_info)

    def reset(self, state=None):
        env_mask = np.zeros(dtype=bool)
        env_mask[0] = True
        return self.reset_all(env_mask, state)

    def step(self, action):
        env_mask = np.zeros(dtype=bool)
        env_mask[0] = True
        return self.step_all(env_mask, action)

    def step_all(self, env_mask, action):
        raise NotImplementedError

    def reset_all(self, env_mask, state=None):
        raise NotImplementedError
