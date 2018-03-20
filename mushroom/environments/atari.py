import gym
from PIL import Image

from mushroom.environments import Environment, MDPInfo
from mushroom.utils.spaces import *


class Atari(Environment):
    """
    The Atari environment as presented in:
    "Human-level control through deep reinforcement learning". Mnih et. al..
    2015.

    """
    def __init__(self, name, width=84, height=84, ends_at_life=False):
        """
        Constructor.

        Args:
             name (str): id name of the Atari game in Gym;
             width (int, 84): width of the screen;
             height (int, 84): height of the screen;
             ends_at_life (bool, False): whether the episode ends when a life is
                lost or not.

        """
        self.__name__ = name

        # MPD creation
        self.env = gym.make(self.__name__)

        # MDP parameters
        self.img_size = (width, height)
        self._episode_ends_at_life = ends_at_life
        self._max_lives = self.env.env.ale.lives()
        self._lives = self._max_lives

        # MDP properties
        action_space = Discrete(self.env.action_space.n)
        observation_space = Box(
            low=0., high=255., shape=(self.img_size[1], self.img_size[0]))
        horizon = np.inf
        gamma = .99
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super(Atari, self).__init__(mdp_info)

    def reset(self, state=None):
        if self._episode_ends_at_life:
            if self._lives == 0 or self._lives == self._max_lives:
                self._state = self._preprocess_observation(self.env.reset())
                self._lives = self._max_lives
        else:
            self._state = self._preprocess_observation(self.env.reset())

        return self._state

    def step(self, action):
        obs, reward, absorbing, info = self.env.step(action)

        if self._episode_ends_at_life:
            if info['ale.lives'] != self._lives:
                absorbing = True
                self._lives = info['ale.lives']

        self._state = self._preprocess_observation(obs)

        return self._state, reward, absorbing, info

    def render(self, mode='human'):
        self.env.render(mode=mode)

    def set_episode_end(self, ends_at_life):
        """
        Setter.

        Args:
            ends_at_life (bool): whether the episode ends when a life is
                lost or not.

        """
        self._episode_ends_at_life = ends_at_life

    def _preprocess_observation(self, obs):
        image = Image.fromarray(obs, 'RGB').convert('L').resize(self.img_size)

        return np.asarray(image.getdata(), dtype=np.uint8).reshape(
            image.size[1], image.size[0])  # Convert to array and return
