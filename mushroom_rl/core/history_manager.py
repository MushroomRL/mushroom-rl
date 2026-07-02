from mushroom_rl.core.mushroom_object import MushroomObject


class HistoryManager(MushroomObject):
    """
    Object in charge of assembling the policy input from the recent observation history, i.e. the context window.

    The observation history is a deterministic function of the observed trajectory, hence it is always reconstructable
    from the stored transitions and is not part of the (latent) policy state. The manager keeps its own rolling buffer
    of the last ``history_length - 1`` observations, which it resets at the beginning of each episode and uses to
    assemble the stacked observation fed to the policy. The same assembly rule is meant to be reused offline (e.g. by
    the replay memory), guaranteeing that the policy input matches between rollout and learning.

    """
    def __init__(self, history_length, obs_shape, obs_dtype, env_backend, agent_backend):
        """
        Constructor.

        Args:
            history_length (int): number of observations stacked as policy input;
            obs_shape (tuple): the shape of the observation;
            obs_dtype: the data type of the observation;
            env_backend (ArrayBackend): the array backend used by the environment;
            agent_backend (ArrayBackend): the array backend used by the agent.

        """
        self._history_length = history_length
        self._obs_shape = obs_shape
        self._obs_dtype = obs_dtype
        self._env_backend = env_backend
        self._agent_backend = agent_backend

        self._buffer = None
        self._vectorized = False

        self._add_save_attr(
            _history_length='primitive',
            _obs_shape='primitive',
            _obs_dtype='primitive',
            _env_backend='primitive',
            _agent_backend='primitive',
            _buffer='none',
            _vectorized='primitive'
        )

    @property
    def history_length(self):
        """
        The number of observations stacked as policy input.

        """
        return self._history_length

    def reset(self):
        """
        Reset the buffer at the beginning of a single-environment episode.

        """
        self._vectorized = False
        self._buffer = self._env_backend.zeros(self._history_length - 1, *self._obs_shape, dtype=self._obs_dtype)

    def reset_vectorized(self, start_mask):
        """
        Reset the buffer for the environments selected by ``start_mask``, leaving the others untouched.

        Args:
            start_mask: boolean mask selecting the environments that are starting a new episode.

        """
        self._vectorized = True
        n_envs = len(start_mask)
        if self._buffer is None or self._buffer.shape[0] != n_envs:
            self._buffer = self._env_backend.zeros(n_envs, self._history_length - 1, *self._obs_shape,
                                                   dtype=self._obs_dtype)
        else:
            self._buffer[start_mask] = 0

    def __call__(self, state):
        """
        Append ``state`` to the buffer and return the stacked observation window.

        Args:
            state: the current observation, already converted to the agent backend.

        Returns:
            The stacked observation, with shape ``(history_length, *obs_shape)`` in the single-environment case and
            ``(n_envs, history_length, *obs_shape)`` in the vectorized case.

        """
        buffer = self._agent_backend.convert_to_backend(self._env_backend, self._buffer)

        if self._vectorized:
            stacked = self._agent_backend.concatenate([buffer, state[:, None]], dim=1)
            new_buffer = stacked[:, 1:]
        else:
            stacked = self._agent_backend.concatenate([buffer, state[None]], dim=0)
            new_buffer = stacked[1:]

        self._buffer = self._env_backend.convert_to_backend(self._agent_backend, new_buffer)

        return stacked
