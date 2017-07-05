import gym
import numpy as np
from PIL import Image

from PyPi.environments import Environment
from PyPi.utils import spaces


class Atari(Environment):
    def __init__(self, name):
        self.__name__ = name

        # MPD creation
        self.env = gym.make(self.__name__)

        # MDP spaces
        self.img_size = (84, 110)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # MDP parameters
        self.horizon = 100
        self.gamma = 0.95

        super(Atari, self).__init__()

    def reset(self, state=None):
        state = self._preprocess_observation(self.env.reset())
        self.env.state = np.array([state, state, state, state])

        return self.get_state()

    def step(self, action):
        _, reward, absorbing, info = self.env.step(action)

        return self.get_state(), reward, absorbing, info

    def render(self, mode='human', close=False):
        self.env.render(mode=mode, close=close)

    def _preprocess_observation(self, obs):
        image = Image.fromarray(obs, 'RGB').convert('L').resize(self.img_size)

        return np.asarray(image.getdata(), dtype=np.uint8).reshape(
            image.size[1], image.size[0])  # Convert to array and return
