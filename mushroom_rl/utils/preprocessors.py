import pickle
import numpy as np

from mushroom_rl.utils.running_stats import RunningStandardization


class NormalizationProcessor(object):
    """
    Normalizes observations from the environment using
        RunningStandardization.

    """
    def __init__(self, mdp_info, clip_obs=10., alpha=1e-32):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): mdp_info object.
            clip_obs (float): Values to clip the normalized
                observations.
            alpha (float): Moving average catchup parameter for
                the normalization.

        """

        self.clip_obs = clip_obs
        self.obs_runstand = RunningStandardization(shape=mdp_info.observation_space.shape,
                                                   alpha=alpha)

    def __call__(self, obs):
        """
        Calls function to normalize the observation.
        Args:
            obs (np.array): observation to be normalized

        Returns a normalized observation array with the same shape

        """
        self.obs_runstand.update_stats(obs)
        norm_obs = np.clip((obs - self.obs_runstand.mean) / self.obs_runstand.std,
                           -self.clip_obs, self.clip_obs)
        return norm_obs

    def get_state(self):
        """
        Returns a dict with the normalization state.

        """
        return self.obs_runstand.get_state()

    def set_state(self, data):
        """
        Sets the current normalization state from the data dict.

        """
        self.obs_runstand.set_state(data)

    def save_state(self, path):
        """
        Saves the running normalization state to path.

        Args:
            path (string): Path to save the running normalization state.

        """
        with open(path, 'wb') as f:
            pickle.dump(self.get_state(), f, protocol=3)

    def load_state(self, path):
        """
        Loads the running normalization state from path.

        Args:
            path (string): Path to load the running normalization
                state from.

        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.set_state(data)


class NormalizationBoxedProcessor(NormalizationProcessor):
    """
    Normalizes observations from the environment using
        mdp_info.observation_space Box range. For observations which
        are not limited falls back to using RunningStandardization.

    """
    def __init__(self, mdp_info, clip_obs=10., alpha=1e-32):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): mdp_info object.
            clip_obs (float): Values to clip the normalized
                observations.
            alpha (float): Moving average catchup parameter for
                the normalization.

        """

        super(NormalizationBoxedProcessor, self).__init__(mdp_info, clip_obs, alpha)

        # create mask where observations will be normalized between
        # boxed values(not inf)
        obs_low, obs_high = (mdp_info.observation_space.low.copy(),
                             mdp_info.observation_space.high.copy())

        self.stand_obs_mask = np.where(~(np.isinf(obs_low) | np.isinf(obs_high)))

        # turn off running stats if all observations will be boxed
        self.run_norm_obs = len(np.squeeze(self.stand_obs_mask)) != obs_low.shape[0]

        # mean/delta values to use for normalization for observations
        self.obs_mean = np.zeros_like(obs_low)
        self.obs_delta = np.ones_like(obs_low)
        self.obs_mean[self.stand_obs_mask] = (obs_high[self.stand_obs_mask]
                                              + obs_low[self.stand_obs_mask]) / 2.0
        self.obs_delta[self.stand_obs_mask] = (obs_high[self.stand_obs_mask]
                                               - obs_low[self.stand_obs_mask]) / 2.0

    def __call__(self, obs):
        """
        Calls function to normalize the observation.

        Args:
            obs (np.array): observation to be normalized

        Returns a normalized observation array with the same shape

        """
        orig_obs = obs.copy()

        if self.run_norm_obs:
            obs = super(NormalizationBoxedProcessor, self).__call__(obs)

        obs[self.stand_obs_mask] = \
            ((orig_obs - self.obs_mean) / self.obs_delta)[self.stand_obs_mask]

        return obs
