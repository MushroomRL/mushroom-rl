import numpy as np

from .environment import Environment


class VectorizedEnvironment(Environment):
    """
    Basic interface used by any MushroomRL vectorized environment.

    """
    def __init__(self, mdp_info, n_envs):
        self._n_envs = n_envs
        self._default_env = 0

        super().__init__(mdp_info)

    def reset(self, state=None):
        env_mask = np.zeros(self._n_envs, dtype=bool)
        env_mask[self._default_env] = True
        return self.reset_all(env_mask, state)

    def step(self, action):
        env_mask = np.zeros(self._n_envs, dtype=bool)
        env_mask[self._default_env] = True
        return self.step_all(env_mask, action)

    def render(self, record=False):
        env_mask = np.zeros(self._n_envs, dtype=bool)
        env_mask[self._default_env] = True

        return self.render_all(env_mask, record=record)

    def reset_all(self, env_mask, state=None):
        """
        Reset all the specified environments to the initial state.

        Args:
            env_mask: mask specifying which environments needs reset.
            state: set of initial states to impose to the environment.

        Returns:
            The initial states of all environments and a listy of episode info dictionaries

        """
        raise NotImplementedError

    def step_all(self, env_mask, action):
        """
        Move all the specified agents from their current state according to the actions.

        Args:
            env_mask: mask specifying which environments needs reset.
            action: set of actions to execute.

        Returns:
            The initial states of all environments and a listy of step info dictionaries

        """
        raise NotImplementedError

    def render_all(self, env_mask, record=False):
        """
        Render all the specified environments to screen.

        Args:
            record (bool, False): whether the visualized images should be returned or not.

        Returns:
            The visualized images, or None if the record flag is set to false.

        """
        raise NotImplementedError

    def set_default_env(self, id):
        """
        Select the id of the default environment that will be executed with the default env interface.

        Args:
            id (int): the number of the selected environment

        """
        assert id < self._n_envs, "The selected ID is higher than the available ones"

        self._default_env = id

    @property
    def number(self):
        return self._n_envs
