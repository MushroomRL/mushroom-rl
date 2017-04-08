import numpy as np
import gym

from PIL import Image

from PyPi.utils import spaces


class Atari(gym.Env):
    """
    The Atari environment.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, game='PongDeterministic-v3'):
        self.game = game
        self.__name__ = game

        # MDP creation
        self.env = gym.make(game)

        # MDP spaces
        self.action_space = spaces.Discrete(self.env.action_space.n)
        self.IMG_SIZE = (84, 110)
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(4,
                                                   self.IMG_SIZE[1],
                                                   self.IMG_SIZE[0]))

        # MDP parameters
        self.horizon = np.inf
        self.gamma = 0.99

        # MDP initialization
        self.env.seed()
        self.reset()

    def reset(self, state=None):
        state = self._preprocess_observation(self.env.reset())
        self.env.state = np.array([state, state, state, state])

        return self.get_state()

    def step(self, action):
        current_state = self.get_state()
        obs, reward, done, info = self.env.step(int(action))

        reward = np.clip(reward, -1, 1)

        obs = self._preprocess_observation(obs)
        self.env.state = self._get_next_state(current_state, obs)

        return self.get_state(), reward, done, info

    def get_state(self):
        return self.env.state

    def _preprocess_observation(self, obs):
        image = Image.fromarray(obs, 'RGB').convert('L').resize(self.IMG_SIZE)
        return np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[1],
                                                                   image.size[0])  # Convert to array and return

    def _get_next_state(self, current, obs):
        # Next state is composed by the last 3 images of the previous state and the new observation
        return np.append(current[1:], [obs], axis=0)

    def render(self, mode='human', close=False):
        self.env.render(mode=mode, close=close)

    def __str__(self):
        return self.game
