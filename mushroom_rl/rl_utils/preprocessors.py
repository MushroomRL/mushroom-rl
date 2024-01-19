import numpy as np

from mushroom_rl.core import Serializable, ArrayBackend
from mushroom_rl.rl_utils.running_stats import RunningStandardization


class Preprocessor(Serializable):
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


class StandardizationPreprocessor(Preprocessor):
    """
    Preprocess observations from the environment using a running
    standardization.

    """
    def __init__(self, mdp_info, backend, clip_obs=10., alpha=1e-32):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information of the MDP;
            backend (str): name of the backend to be used;
            clip_obs (float, 10.): values to clip the normalized observations;
            alpha (float, 1e-32): moving average catchup parameter for the
                normalization.

        """
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
        assert obs.shape == self._obs_shape, \
            "Values given to running_norm have incorrect shape " \
            "(obs shape: {},  expected shape: {})" \
            .format(obs.shape, self._obs_shape)

        norm_obs = self._array_backend.clip(
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
    def __init__(self, mdp_info, backend, clip_obs=10., alpha=1e-32):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information of the MDP;
            backend (str): name of the backend to be used;
            clip_obs (float, 10.): values to clip the normalized observations;
            alpha (float, 1e-32): moving average catchup parameter for the
                normalization.

        """
        super(MinMaxPreprocessor, self).__init__(mdp_info, backend, clip_obs, alpha)

        obs_low, obs_high = (self._array_backend.convert(mdp_info.observation_space.low.copy(),
                                                         mdp_info.observation_space.high.copy()))

        self._obs_mask = self._array_backend.where((self._array_backend.abs(obs_low) < 1e20) &
                                                   (self._array_backend.abs(obs_high) < 1e20))

        self._obs_mask = self._array_backend.concatenate(self._obs_mask)

        assert self._obs_mask.sum() > 0, "All observations have unlimited/extremely large range, " \
                                         "you should use StandardizationPreprocessor instead."

        self._run_norm_obs = len(self._array_backend.squeeze(self._obs_mask)) != obs_low.shape[0]

        self._obs_mean = self._array_backend.zeros_like(obs_low)
        self._obs_delta = self._array_backend.ones_like(obs_low)
        self._obs_mean[self._obs_mask] = (obs_high[self._obs_mask] + obs_low[self._obs_mask]) / 2.
        self._obs_delta[self._obs_mask] = (obs_high[self._obs_mask] - obs_low[self._obs_mask]) / 2.

        self._add_save_attr(
            _array_backend='pickle',
            _run_norm_obs='primitive',
            _obs_mask='numpy',
            _obs_mean='numpy',
            _obs_delta='numpy'
        )

    def __call__(self, obs):
        orig_obs = self._array_backend.copy(obs)

        if self._run_norm_obs:
            obs = super(MinMaxPreprocessor, self).__call__(obs)

        obs[self._obs_mask] = \
            ((orig_obs - self._obs_mean) / self._obs_delta)[self._obs_mask]

        return obs
