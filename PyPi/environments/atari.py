import gym
from PIL import Image

from PyPi.environments import Environment
from PyPi.utils.spaces import *


class Atari(Environment):
    def __init__(self, name, ends_at_life=False):
        self.__name__ = name

        # MPD creation
        self.env = gym.make(self.__name__)

        # MDP spaces
        self.img_size = (84, 84)
        self.action_space = Discrete(self.env.action_space.n)
        self.observation_space = Box(
            low=0., high=255., shape=(self.img_size[1], self.img_size[0], 3))

        # MDP parameters
        self.horizon = np.inf
        self.gamma = 0.99

        # MDP properties
        self._episode_ends_at_life = ends_at_life
        self._max_lives = self.env.env.ale.lives()
        self._lives = 0

        super(Atari, self).__init__()

    def reset(self, state=None):
        if self._episode_ends_at_life:
            if self._lives == 0:
                state = self._preprocess_observation(self.env.reset())
                self._state = np.array([state, state, state, state])
                self._lives = self._max_lives
        else:
            state = self._preprocess_observation(self.env.reset())
            self._state = np.array([state, state, state, state])

        return self._state

    def step(self, action):
        obs, reward, absorbing, info = self.env.step(action)

        if self._episode_ends_at_life:
            if info['ale.lives'] != self._lives:
                absorbing = True
                self._lives = info['ale.lives']

        obs = self._preprocess_observation(obs)
        self._state = self._get_next_state(obs)

        return self._state, reward, absorbing, info

    def render(self, mode='human', close=False):
        self.env.render(mode=mode, close=close)

    def set_episode_end(self, ends_at_life):
        self._episode_ends_at_life = ends_at_life

    def _get_next_state(self, obs):
        return np.append(self._state[1:], [obs], axis=0)

    def _preprocess_observation(self, obs):
        image = Image.fromarray(obs, 'RGB').convert('L').resize(self.img_size)

        return np.asarray(image.getdata(), dtype=np.uint8).reshape(
            image.size[1], image.size[0])  # Convert to array and return
