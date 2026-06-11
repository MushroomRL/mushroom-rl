import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import cv2
import gymnasium

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.environments import Gymnasium
from mushroom_rl.rl_utils.spaces import Discrete, Box


class MiniGrid(Gymnasium):
    """
    Interface for gym_minigrid environments. It makes it possible to
    use all MiniGrid environments that do not use text instructions, such as
    MultiRoom, KeyCorridor, BlockedUnblockPickup, ObstructedMaze.
    This environment uses either MiniGrid's default 7x7x3 observations or their
    pixel 56x56x3 version. In both cases, the state is partially observable.

    """
    def __init__(self, name, horizon=None, gamma=0.99, fixed_seed=None, use_pixels=False):
        """
        Constructor.

        Args:
             name (str): name of the environment;
             horizon (int, None): the horizon;
             gamma (float, 0.99): the discount factor;
             fixed_seed (int, None): if passed, it fixes the seed of the
                environment at every reset. This way, the environment is fixed
                rather than procedurally generated;
             use_pixels (bool, False): if True, MiniGrid's default 7x7x3
                observations is converted to an image of resolution 56x56x3.

        """
        self._not_pybullet = True
        self._first = True

        env = gymnasium.make(name)
        obs_high = 10.
        if use_pixels:
            env = RGBImgPartialObsWrapper(env)
            obs_high = 255.
        env = ImgObsWrapper(env)
        self.env = env

        self._fixed_seed = fixed_seed
        self._img_size = env.observation_space.shape[0:2]

        if horizon is None:
            horizon = self.env.unwrapped.max_steps

        action_space = Discrete(self.env.action_space.n)
        observation_space = Box(low=0., high=obs_high, shape=self._img_size)
        self.env.unwrapped.max_steps = horizon + 1
        dt = 1 / self.env.unwrapped.metadata["render_fps"]
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)

        Environment.__init__(self, mdp_info)

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

    def _preprocess_frame(self, obs):
        image = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, self._img_size, interpolation=cv2.INTER_LINEAR)
        return np.array(image, dtype=np.uint8)
