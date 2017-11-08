import gym
from gym.utils import seeding


class MDPInfo:
    def __init__(self, observation_space, action_space, gamma, horizon):
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.horizon = horizon

    @property
    def size(self):
        return self.observation_space.size + self.action_space.size

    @property
    def shape(self):
        return self.observation_space.shape + self.action_space.shape


class Environment(gym.Env):
    def __init__(self, mdp_info):
        # MDP initialization
        self._seed()
        self.reset()

        self._mdp_info = mdp_info

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def info(self):
        return self._mdp_info

    def __str__(self):
        return self.__name__
