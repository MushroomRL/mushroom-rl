import pickle
import numpy as np

from mushroom_rl.utils.running_stats import RunningStandardization


class StandardizationPreprocessor(object):
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
        self.clip_obs = clip_obs
        self.obs_shape = mdp_info.observation_space.shape
        self.obs_runstand = RunningStandardization(shape=self.obs_shape,
                                                   alpha=alpha)

    def __call__(self, obs):
        """
        Call function to normalize the observation.

        Args:
            obs (np.ndarray): observation to be normalized.

        Returns:
            Normalized observation array with the same shape.

        """
        assert obs.shape == self.obs_shape, \
            "Values given to running_norm have incorrect shape " \
            "(obs shape: {},  expected shape: {})" \
            .format(obs.shape, self.obs_shape)

        self.obs_runstand.update_stats(obs)
        norm_obs = np.clip(
            (obs - self.obs_runstand.mean) / self.obs_runstand.std,
            -self.clip_obs, self.clip_obs
        )

        return norm_obs

    def get_state(self):
        """
        Returns:
            A dictionary with the normalization state.

        """
        return self.obs_runstand.get_state()

    def set_state(self, data):
        """
        Set the current normalization state from the data dict.

        """
        self.obs_runstand.set_state(data)

    def save_state(self, path):
        """
        Save the running normalization state to path.

        Args:
            path (str): path to save the running normalization state.

        """
        with open(path, 'wb') as f:
            pickle.dump(self.get_state(), f, protocol=3)

    def load_state(self, path):
        """
        Load the running normalization state from path.

        Args:
            path (string): path to load the running normalization state from.

        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.set_state(data)


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

        self.stand_obs_mask = np.where(
            np.logical_and(np.abs(obs_low) < 1e20, np.abs(obs_low) < 1e20)
        )

        assert np.squeeze(self.stand_obs_mask).size > 0, \
            "All observations have unlimited/extremely large range," \
            " you should use StandardizationPreprocessor instead."

        self.run_norm_obs = len(np.squeeze(self.stand_obs_mask)) != obs_low.shape[0]

        self.obs_mean = np.zeros_like(obs_low)
        self.obs_delta = np.ones_like(obs_low)
        self.obs_mean[self.stand_obs_mask] = (
            obs_high[self.stand_obs_mask] + obs_low[self.stand_obs_mask]) / 2.
        self.obs_delta[self.stand_obs_mask] = (
            obs_high[self.stand_obs_mask] - obs_low[self.stand_obs_mask]) / 2.

    def __call__(self, obs):
        """
        Call function to normalize the observation.

        Args:
            obs (np.ndarray): observation to be normalized.

        Returns:
            Normalized observation array with the same shape.

        """
        orig_obs = obs.copy()

        if self.run_norm_obs:
            obs = super(MinMaxPreprocessor, self).__call__(obs)

        obs[self.stand_obs_mask] = \
            ((orig_obs - self.obs_mean) / self.obs_delta)[self.stand_obs_mask]

        return obs
