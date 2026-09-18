from mushroom_rl.core.environment import Environment
from mushroom_rl.core.array_backend import ArrayBackend


class VectorizedEnvironment(Environment):
    """
    Basic interface used by any MushroomRL vectorized environment.

    """
    def __init__(self, mdp_info, n_envs):
        self._n_envs = n_envs
        self._default_env = 0

        super().__init__(mdp_info)

    def reset(self, state=None):
        """
        Reset the state of the default environment, leaving the other copies untouched.

        Args:
            state (Array, None): the optional initial state to impose to the default environment.

        Returns:
            The initial state of the default environment, and its episode info dictionary.

        """
        arraybackend = ArrayBackend.get_array_backend(self._mdp_info.backend)
        env_mask = arraybackend.zeros(self._n_envs, dtype=bool, device=self._mdp_info.device)
        env_mask[self._default_env] = True

        if state is not None:
            states = arraybackend.zeros(self._n_envs, *arraybackend.shape(state), device=self._mdp_info.device)
            states[self._default_env] = state
        else:
            states = None

        states, episode_infos = self.reset_all(env_mask, states)

        return states[self._default_env], self._default_env_info(episode_infos)

    def step(self, action):
        """
        Move the default environment from its current state according to the action, leaving the other copies
        untouched.

        Args:
            action (Array): the action to execute in the default environment.

        Returns:
            The state reached by the default environment, the reward obtained, the absorbing flag, and its step
            info dictionary.

        """
        arraybackend = ArrayBackend.get_array_backend(self._mdp_info.backend)
        env_mask = arraybackend.zeros(self._n_envs, dtype=bool, device=self._mdp_info.device)
        env_mask[self._default_env] = True

        actions = arraybackend.zeros(self._n_envs, *arraybackend.shape(action), device=self._mdp_info.device)
        actions[self._default_env] = action

        next_states, rewards, absorbings, step_infos = self.step_all(env_mask, actions)

        return (next_states[self._default_env], rewards[self._default_env], absorbings[self._default_env],
                self._default_env_info(step_infos))

    def render(self, record=False):
        array_backend = ArrayBackend.get_array_backend(self._mdp_info.backend)
        env_mask = array_backend.zeros(self._n_envs, dtype=bool, device=self._mdp_info.device)
        env_mask[self._default_env] = True

        frame = self.render_all(env_mask, record=record)

        if not record:
            return None

        if frame is not None and frame.ndim == 4:
            frame = frame[0]

        return frame

    def reset_all(self, env_mask, state=None):
        """
        Reset all the specified environments to the initial state.

        Args:
            env_mask: mask specifying which environments needs reset.
            state: set of initial states to impose to the environment.

        Returns:
            The initial states of all the selected environments, and a list of episode info dictionaries.

        """
        raise NotImplementedError

    def step_all(self, env_mask, action):
        """
        Move all the specified agents from their current state according to the actions.

        Args:
            env_mask: mask specifying which environments needs to do a step.
            action: set of actions to execute.

        Returns:
            The next states of all the selected environments, the rewards obtained, the absorbing flags, and
            a list of step info dictionaries.

        """
        raise NotImplementedError

    def render_all(self, env_mask, record=False):
        """
        Render all the specified environments to screen.

        Args:
            record (bool, False): whether the visualized images should be returned or not.

        Returns:
            The visualized images of all the selected environments, or None if the record flag is set to
            false.

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

    def _default_env_info(self, info):
        """
        Args:
            info (dict, list): the information of every environment, either as a list of dictionaries, one per
                environment, or as a dictionary of arrays.

        Returns:
            The information of the default environment, as a dictionary.

        """
        if isinstance(info, dict):
            return {key: value[self._default_env] for key, value in info.items()}

        return info[self._default_env]
