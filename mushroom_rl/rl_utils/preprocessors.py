from mushroom_rl.core import MushroomBackendObject
from mushroom_rl.rl_utils.running_stats import RunningStandardization


class Preprocessor(MushroomBackendObject):
    """
    Abstract preprocessor class.

    """
    def __init__(self, backend='numpy', device=None):
        """
        Constructor.

        Args:
            backend (str, 'numpy'): name of the array backend of the observations the preprocessor is applied to;
            device (str, None): device the preprocessor data is stored on, or ``None`` for the default one.

        """
        super().__init__(backend, device)

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


class StandardizationPreprocessor(Preprocessor):
    """
    Preprocess observations from the environment using a running
    standardization.

    """
    def __init__(self, mdp_info, clip_obs=10., alpha=1e-32, backend=None, device=None):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information of the MDP;
            clip_obs (float, 10.): values to clip the normalized observations;
            alpha (float, 1e-32): moving average catchup parameter for the normalization;
            backend (str, None): array backend of the observations the preprocessor is applied to; when
                ``None`` the MDP backend is used, which is the one a core preprocessor receives. An agent
                preprocessor must be given the agent backend instead;
            device (str, None): device the statistics are stored on, defaulting to the MDP one when the backend is
                the MDP one.

        """
        backend = mdp_info.backend_name if backend is None else backend
        device = mdp_info.device if device is None and backend == mdp_info.backend_name else device
        super().__init__(backend, device)

        self._clip_obs = clip_obs
        self._obs_shape = mdp_info.observation_space.shape
        self._obs_runstand = RunningStandardization(shape=self._obs_shape,
                                                    alpha=alpha,
                                                    backend=backend,
                                                    device=self._device)

        self._add_save_attr(
            _clip_obs='primitive',
            _obs_shape='primitive',
            _obs_runstand='mushroom'
        )

    def __call__(self, obs):

        norm_obs = self._backend.clip(
            (obs - self._obs_runstand.mean) / self._obs_runstand.std,
            -self._clip_obs, self._clip_obs
        )

        return norm_obs

    def update(self, obs):
        self._obs_runstand.update_stats(obs)


class MinMaxPreprocessor(StandardizationPreprocessor):
    """
    Preprocess observations from the environment using the bounds of the
    observation space of the environment. For observations that are not limited
    falls back to using running mean standardization.

    """
    def __init__(self, mdp_info, clip_obs=10., alpha=1e-32, backend=None, device=None):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information of the MDP;
            clip_obs (float, 10.): values to clip the normalized observations;
            alpha (float, 1e-32): moving average catchup parameter for the
                normalization;
            backend (str, None): array backend of the observations the preprocessor is applied to; when
                ``None`` the MDP backend is used, which is the one a core preprocessor receives. An agent
                preprocessor must be given the agent backend instead;
            device (str, None): device the statistics are stored on, defaulting to the MDP one when the backend is
                the MDP one.

        """
        super().__init__(mdp_info, clip_obs, alpha, backend, device)

        obs_low, obs_high = self._backend.convert(mdp_info.observation_space.low,
                                                  mdp_info.observation_space.high, device=self._device)

        self._obs_mask = (self._backend.abs(obs_low) < 1e20) & (self._backend.abs(obs_high) < 1e20)

        assert self._obs_mask.sum() > 0, "All observations have unlimited/extremely large range, " \
                                         "you should use StandardizationPreprocessor instead."

        self._run_norm_obs = not bool(self._obs_mask.all())

        self._obs_mean = self._backend.zeros_like(obs_low)
        self._obs_delta = self._backend.ones_like(obs_low)
        self._obs_mean[self._obs_mask] = (obs_high[self._obs_mask] + obs_low[self._obs_mask]) / 2.
        self._obs_delta[self._obs_mask] = (obs_high[self._obs_mask] - obs_low[self._obs_mask]) / 2.

        self._add_save_attr(
            _run_norm_obs='primitive',
            _obs_mask=self._backend.get_backend_serialization(),
            _obs_mean=self._backend.get_backend_serialization(),
            _obs_delta=self._backend.get_backend_serialization()
        )

    def __call__(self, obs):
        bounded = self._obs_mask
        norm_obs = super().__call__(obs) if self._run_norm_obs else self._backend.copy(obs)
        norm_obs[..., bounded] = (obs[..., bounded] - self._obs_mean[..., bounded]) / self._obs_delta[..., bounded]

        return norm_obs
