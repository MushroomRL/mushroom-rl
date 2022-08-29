import numpy as np

from mushroom_rl.core import Serializable
from mushroom_rl.utils.running_stats import RunningStandardization


class StandardizationPreprocessor(Serializable):
    """
    Preprocess observations from the environment using a running
    standardization.

    """
    def __init__(self, mdp_info, clip_obs=10., alpha=1e-32):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information of the MDP;
            clip_obs (float, 10.): values to clip the normalized observations;
            alpha (float, 1e-32): moving average catchup parameter for the
                normalization.

        """
        self._clip_obs = clip_obs
        self._obs_shape = mdp_info.observation_space.shape
        self._obs_runstand = RunningStandardization(shape=self._obs_shape,
                                                    alpha=alpha)

        self._add_save_attr(
            _clip_obs='primitive',
            _obs_shape='primitive',
            _obs_runstand='mushroom'
        )

    def __call__(self, obs):
        """
        Call function to normalize the observation.

        Args:
            obs (np.ndarray): observation to be normalized.

        Returns:
            Normalized observation array with the same shape.

        """
        assert obs.shape == self._obs_shape, \
            "Values given to running_norm have incorrect shape " \
            "(obs shape: {},  expected shape: {})" \
            .format(obs.shape, self._obs_shape)

        self._obs_runstand.update_stats(obs)
        norm_obs = np.clip(
            (obs - self._obs_runstand.mean) / self._obs_runstand.std,
            -self._clip_obs, self._clip_obs
        )

        return norm_obs


class MinMaxPreprocessor(StandardizationPreprocessor):
    """
    Preprocess observations from the environment using the bounds of the
    observation space of the environment. For observations that are not limited
    falls back to using running mean standardization.

    """
    def __init__(self, mdp_info, clip_obs=10., alpha=1e-32):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information of the MDP;
            clip_obs (float, 10.): values to clip the normalized observations;
            alpha (float, 1e-32): moving average catchup parameter for the
                normalization.

        """
        super(MinMaxPreprocessor, self).__init__(mdp_info, clip_obs,
                                                 alpha)

        obs_low, obs_high = (mdp_info.observation_space.low.copy(),
                             mdp_info.observation_space.high.copy())

        self._obs_mask = np.where(
            np.logical_and(np.abs(obs_low) < 1e20, np.abs(obs_high) < 1e20)
        )

        assert np.squeeze(self._obs_mask).size > 0, \
            "All observations have unlimited/extremely large range," \
            " you should use StandardizationPreprocessor instead."

        self._run_norm_obs = len(np.squeeze(self._obs_mask)) != obs_low.shape[0]

        self._obs_mean = np.zeros_like(obs_low)
        self._obs_delta = np.ones_like(obs_low)
        self._obs_mean[self._obs_mask] = (obs_high[self._obs_mask] + obs_low[self._obs_mask]) / 2.
        self._obs_delta[self._obs_mask] = (obs_high[self._obs_mask] - obs_low[self._obs_mask]) / 2.

        self._add_save_attr(
            _run_norm_obs='primitive',
            _obs_mask='numpy',
            _obs_mean='numpy',
            _obs_delta='numpy'
        )

    def __call__(self, obs):
        """
        Call function to normalize the observation.

        Args:
            obs (np.ndarray): observation to be normalized.

        Returns:
            Normalized observation array with the same shape.

        """
        orig_obs = obs.copy()

        if self._run_norm_obs:
            obs = super(MinMaxPreprocessor, self).__call__(obs)

        obs[self._obs_mask] = \
            ((orig_obs - self._obs_mean) / self._obs_delta)[self._obs_mask]

        return obs
