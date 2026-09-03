from mushroom_rl.core import MushroomObject, ArrayBackend
from mushroom_rl.rl_utils.running_stats import RunningStandardization


class Preprocessor(MushroomObject):
    """
    Abstract preprocessor class.

    """
    def __call__(self, obs):
        """
        Preprocess the observations.

        Args:
            obs (Array): observations to be preprocessed.

        Return:
            Preprocessed observations.

        """
        # TODO: Support vectorized environment and batch preprocessing.
        raise NotImplementedError

    def update(self, obs):
        """
        Update internal state of the preprocessor using the current observations.

        Args:
            obs (Array): observations to be preprocessed.

        """
        # TODO: Support vectorized environment and batch update.
        pass

    @property
    def backend(self):
        """
        Returns:
            The name of the array backend the preprocessor operates in, or ``None`` when it accepts any.

        """
        return None


class StandardizationPreprocessor(Preprocessor):
    """
    Preprocess observations from the environment using a running
    standardization.

    """
    def __init__(self, mdp_info, clip_obs=10., alpha=1e-32, backend=None):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information of the MDP;
            clip_obs (float, 10.): values to clip the normalized observations;
            alpha (float, 1e-32): moving average catchup parameter for the normalization;
            backend (str, None): array backend of the observations the preprocessor is applied to; when
                ``None`` the MDP backend is used, which is the one a core preprocessor receives. An agent
                preprocessor must be given the agent backend instead.

        """
        backend = mdp_info.backend if backend is None else backend
        self._clip_obs = clip_obs
        self._obs_shape = mdp_info.observation_space.shape
        self._array_backend = ArrayBackend.get_array_backend(backend)
        self._obs_runstand = RunningStandardization(shape=self._obs_shape,
                                                    backend=backend,
                                                    alpha=alpha)

        self._add_save_attr(
            _clip_obs='primitive',
            _obs_shape='primitive',
            _array_backend='pickle',
            _obs_runstand='mushroom'
        )

    def __call__(self, obs):

        norm_obs = self._array_backend.clip(
            (obs - self._obs_runstand.mean) / self._obs_runstand.std,
            -self._clip_obs, self._clip_obs
        )

        return norm_obs

    def update(self, obs):
        self._obs_runstand.update_stats(obs)

    @property
    def backend(self):
        return self._array_backend.get_backend_name()


class MinMaxPreprocessor(StandardizationPreprocessor):
    """
    Preprocess observations from the environment using the bounds of the
    observation space of the environment. For observations that are not limited
    falls back to using running mean standardization.

    """
    def __init__(self, mdp_info, clip_obs=10., alpha=1e-32, backend=None):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information of the MDP;
            clip_obs (float, 10.): values to clip the normalized observations;
            alpha (float, 1e-32): moving average catchup parameter for the
                normalization;
            backend (str, None): array backend of the observations the preprocessor is applied to; when
                ``None`` the MDP backend is used, which is the one a core preprocessor receives. An agent
                preprocessor must be given the agent backend instead.

        """
        super().__init__(mdp_info, clip_obs, alpha, backend)

        obs_low, obs_high = self._array_backend.convert(mdp_info.observation_space.low,
                                                        mdp_info.observation_space.high)

        self._obs_mask = (self._array_backend.abs(obs_low) < 1e20) & (self._array_backend.abs(obs_high) < 1e20)

        assert self._obs_mask.sum() > 0, "All observations have unlimited/extremely large range, " \
                                         "you should use StandardizationPreprocessor instead."

        self._run_norm_obs = not bool(self._obs_mask.all())

        self._obs_mean = self._array_backend.zeros_like(obs_low)
        self._obs_delta = self._array_backend.ones_like(obs_low)
        self._obs_mean[self._obs_mask] = (obs_high[self._obs_mask] + obs_low[self._obs_mask]) / 2.
        self._obs_delta[self._obs_mask] = (obs_high[self._obs_mask] - obs_low[self._obs_mask]) / 2.

        self._add_save_attr(
            _array_backend='pickle',
            _run_norm_obs='primitive',
            _obs_mask=self._array_backend.get_backend_serialization(),
            _obs_mean=self._array_backend.get_backend_serialization(),
            _obs_delta=self._array_backend.get_backend_serialization()
        )

    def __call__(self, obs):
        bounded = self._obs_mask
        norm_obs = super().__call__(obs) if self._run_norm_obs else self._array_backend.copy(obs)
        norm_obs[..., bounded] = (obs[..., bounded] - self._obs_mean[..., bounded]) / self._obs_delta[..., bounded]

        return norm_obs
