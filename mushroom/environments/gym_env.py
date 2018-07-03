import gym

from gym import spaces as gym_spaces
from mushroom.environments import Environment, MDPInfo
from mushroom.utils.spaces import *


class Gym(Environment):
    """
    Interface for OpenAI Gym environments. It makes it possible to use every
    Gym environment just providing the id, except for the Atari games that
    are managed in a separate class.

    """
    def __init__(self, name, horizon, gamma):
        """
        Constructor.

        Args:
             name (str): gym id of the environment;
             horizon (int): horizon of the MDP;
             horizon (int): the horizon;
             gamma (float): the discount factor.

        """
        # MPD creation
        self.env = gym.make(name)

        self.env._max_episode_steps = np.inf  # Hack to ignore gym time limit.

        # MDP properties
        assert not isinstance(self.env.observation_space,
                              gym_spaces.MultiDiscrete)
        assert not isinstance(self.env.action_space, gym_spaces.MultiDiscrete)

        action_space = self._convert_gym_space(self.env.action_space)
        observation_space = self._convert_gym_space(self.env.observation_space)
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        if isinstance(action_space, Discrete):
            self._convert_action = self._convert_action_function
        else:
            self._convert_action = self._no_convert

        if isinstance(observation_space,
                      Discrete) and len(observation_space.size) > 1:
                self._convert_state = self._convert_state_function
        else:
            self._convert_state = self._no_convert

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            self._state = self.env.reset()
        else:
            self.env.reset()
            self.env.state = state
            self._state = state

        return self._state

    def step(self, action):
        action = self._convert_action(action)
        self._state, reward, absorbing, info = self.env.step(action)

        self._state = self._convert_state(self._state)

        return self._state, reward, absorbing, info

    def render(self, mode='human'):
        self.env.render(mode=mode)

    def stop(self):
        #self.env.close()
        pass

    @staticmethod
    def _convert_gym_space(space):
        if isinstance(space, gym_spaces.Discrete):
            return Discrete(space.n)
        elif isinstance(space, gym_spaces.Box):
            return Box(low=space.low, high=space.high, shape=space.shape)
        else:
            raise ValueError

    @staticmethod
    def _no_convert(x):
        return x

    @staticmethod
    def _convert_state_function(state):
        return state - state.low

    @staticmethod
    def _convert_action_function(action):
        return int(action[0])
