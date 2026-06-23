import cv2
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

from mushroom_rl.environments import Gymnasium
from mushroom_rl.core.spaces import Box


class MiniGridBase(Gymnasium):
    def __init__(self, name, horizon, gamma, fixed_seed, headless, wrappers, obs_high):
        super().__init__(name, horizon=horizon, gamma=gamma, headless=headless, wrappers=wrappers)

        self._fixed_seed = fixed_seed
        self._img_size = self.env.observation_space.shape[0:2]
        self.info.observation_space = Box(low=0., high=obs_high, shape=self._obs_shape())
        self.info.dt = 1 / self.env.unwrapped.metadata["render_fps"]
        self.env.unwrapped.max_steps = self.info.horizon + 1

    def reset(self, state=None):
        obs, info = self.env.reset(seed=self._fixed_seed)
        return self._preprocess_frame(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action[0])
        reward = float(reward)
        if reward > 0:
            reward = 1.

        absorbing = terminated or truncated

        return self._preprocess_frame(obs), reward, absorbing, info

    def _obs_shape(self):
        raise NotImplementedError

    def _preprocess_frame(self, obs):
        raise NotImplementedError


class MiniGrid(MiniGridBase):
    """
    Interface for MiniGrid environments using the symbolic 7x7x3 observation.
    Each cell is encoded as (object_type, color, state), returned as a (3, H, W)
    array. Suitable for environments where color and door state matter.

    """
    def __init__(self, name, horizon=None, gamma=0.99, fixed_seed=None, headless=False):
        """
        Constructor.

        Args:
             name (str): name of the environment;
             horizon (int, None): the horizon;
             gamma (float, 0.99): the discount factor;
             fixed_seed (int, None): if passed, fixes the seed at every reset;
             headless (bool, False): if True, the rendering is forced to be headless.

        """
        super().__init__(name, horizon, gamma, fixed_seed, headless,
                         wrappers=[ImgObsWrapper], obs_high=10.)

    def _obs_shape(self):
        return 3, *self._img_size

    def _preprocess_frame(self, obs):
        return np.ascontiguousarray(obs.transpose(2, 0, 1), dtype=np.uint8)


class MiniGridRGB(MiniGridBase):
    """
    Interface for MiniGrid environments using pixel observations.
    The 56x56x3 RGB partial observation is converted to grayscale, suitable
    for CNN-based agents such as DQN.

    """
    def __init__(self, name, horizon=None, gamma=0.99, fixed_seed=None, headless=False):
        """
        Constructor.

        Args:
             name (str): name of the environment;
             horizon (int, None): the horizon;
             gamma (float, 0.99): the discount factor;
             fixed_seed (int, None): if passed, fixes the seed at every reset;
             headless (bool, False): if True, the rendering is forced to be headless.

        """
        super().__init__(name, horizon, gamma, fixed_seed, headless,
                         wrappers=[RGBImgPartialObsWrapper, ImgObsWrapper], obs_high=255.)

    def _obs_shape(self):
        return self._img_size

    def _preprocess_frame(self, obs):
        image = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, self._img_size, interpolation=cv2.INTER_LINEAR)
        return np.array(image, dtype=np.uint8)