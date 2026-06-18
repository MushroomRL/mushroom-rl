import gymnasium as gym
import ale_py

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.core.spaces import Box, Discrete
from mushroom_rl.utils.viewer import ImageViewer

import cv2

gym.register_envs(ale_py)

class Atari(Environment):
    """
    The Atari environment as presented in:
    "Human-level control through deep reinforcement learning". Mnih et. al..
    2015.

    """
    def __init__(self,
                 name,
                 width = 84,
                 height = 84,
                 full_action_space = False,
                 repeat_action_probability = 0.25,
                 frameskip = 4,
                 headless = False
                 ):
        """
        Constructor.

        Args:
            name (str): id name of the Atari game in Gymnasium;
            width (int, 84): width of the screen;
            height (int, 84): height of the screen;
            frameskip (int, 4): number of frames to skip (repeat action) per step;
            headless (bool, False): if True, the rendering is forced to be headless.

        """
        assert 'v5' in name, 'This wrapper supports only v5 ALE environments'
        self.env = gym.make(
                            name,
                            full_action_space=full_action_space,
                            frameskip=1,
                            repeat_action_probability=repeat_action_probability,
                            render_mode='rgb_array'
                            )

        self.name = name
        self.state_height, self.state_width = (height, width)
        self.n_skipped_frames = frameskip
        self._headless = headless

        self.original_state_height, self.original_state_width, _ = self.env.observation_space._shape
        self.screen_buffer = [
            np.empty((self.original_state_height, self.original_state_width), dtype=np.uint8),
            np.empty((self.original_state_height, self.original_state_width), dtype=np.uint8),
        ]

        action_space = Discrete(self.env.action_space.n)
        observation_space = Box(low=0., high=255., shape=(self.state_height, self.state_width), data_type=np.uint8)
        horizon = 27_000
        gamma = .99
        dt = 1/60
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)

        self._viewer = ImageViewer((self.original_state_width, self.original_state_height), dt, headless=self._headless)

        self._seed = None
        self.state_ = None

        super().__init__(mdp_info)

    def seed(self, seed):
        self._seed = seed

    def reset(self, state=None):
        _, info = self.env.reset(seed=self._seed)
        self._seed = None

        self.n_steps = 0

        if state is None:
            self.env.unwrapped.ale.getScreenGrayscale(self.screen_buffer[0])
            self.screen_buffer[1].fill(0)

            self.state_ = self.resize()
        else:
            self.state_ = state

        return self.state_, info

    def step(self, action):
        action = action[0]
        reward = 0

        for idx_frame in range(self.n_skipped_frames):
            _, reward_, absorbing, _, info = self.env.step(action)

            reward += reward_

            if idx_frame >= self.n_skipped_frames - 2:
                t = idx_frame - (self.n_skipped_frames - 2)
                self.env.unwrapped.ale.getScreenGrayscale(self.screen_buffer[t])

            if absorbing:
                break

        self.state_ = self.pool_and_resize()

        self.n_steps += 1

        return self.state_, reward, absorbing, info
    
    def pool_and_resize(self) -> np.ndarray:
        np.maximum(self.screen_buffer[0], self.screen_buffer[1], out=self.screen_buffer[0])

        return self.resize()

    def resize(self):
        return np.asarray(
            cv2.resize(self.screen_buffer[0], (self.state_width, self.state_height), interpolation=cv2.INTER_AREA),
            dtype=np.uint8,
        )

    def render(self, record=False):
        img = self.env.render()

        self._viewer.display(img)

        if record:
            return img
        else:
            return None
    
    def stop(self):
        self.env.close()
        self._viewer.close()
