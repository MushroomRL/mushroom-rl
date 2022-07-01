import time

import gym
from gym import spaces as gym_spaces

import numpy as np

try:
    import pybullet_envs
    pybullet_found = True
except ImportError:
    pybullet_found = False

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import *

gym.logger.set_level(40)


class Gym(Environment):
    """
    Interface for OpenAI Gym environments. It makes it possible to use every
    Gym environment just providing the id, except for the Atari games that
    are managed in a separate class.

    """
    def __init__(self, name, horizon=None, gamma=0.99, wrappers=None, wrappers_args=None,
                 **env_args):
        """
        Constructor.

        Args:
             name (str): gym id of the environment;
             horizon (int): the horizon. If None, use the one from Gym;
             gamma (float, 0.99): the discount factor;
             wrappers (list, None): list of wrappers to apply over the environment. It
                is possible to pass arguments to the wrappers by providing
                a tuple with two elements: the gym wrapper class and a
                dictionary containing the parameters needed by the wrapper
                constructor;
            wrappers_args (list, None): list of list of arguments for each wrapper;
            ** env_args: other gym environment parameters.

        """

        # MDP creation
        self._not_pybullet = True
        self._first = True
        if pybullet_found and '- ' + name in pybullet_envs.getList():
            import pybullet
            pybullet.connect(pybullet.DIRECT)
            self._not_pybullet = False

        self.env = gym.make(name, **env_args)

        self._render_dt = self.env.unwrapped.dt if hasattr(self.env.unwrapped, "dt") else 0.0

        if wrappers is not None:
            if wrappers_args is None:
                wrappers_args = [dict()] * len(wrappers)
            for wrapper, args in zip(wrappers, wrappers_args):
                if isinstance(wrapper, tuple):
                    self.env = wrapper[0](self.env, *args, **wrapper[1])
                else:
                    self.env = wrapper(self.env, *args, **env_args)

        horizon = self._set_horizon(self.env, horizon)

        # MDP properties
        assert not isinstance(self.env.observation_space,
                              gym_spaces.MultiDiscrete)
        assert not isinstance(self.env.action_space, gym_spaces.MultiDiscrete)

        action_space = self._convert_gym_space(self.env.action_space)
        observation_space = self._convert_gym_space(self.env.observation_space)
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        if isinstance(action_space, Discrete):
            self._convert_action = lambda a: a[0]
        else:
            self._convert_action = lambda a: a

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            return np.atleast_1d(self.env.reset())
        else:
            self.env.reset()
            self.env.state = state

            return np.atleast_1d(state)

    def step(self, action):
        action = self._convert_action(action)
        obs, reward, absorbing, info = self.env.step(action)

        return np.atleast_1d(obs), reward, absorbing, info

    def render(self, mode='human'):
        if self._first or self._not_pybullet:
            self.env.render(mode=mode)
            self._first = False
            time.sleep(self._render_dt)

    def stop(self):
        try:
            if self._not_pybullet:
                self.env.close()
        except:
            pass

    @staticmethod
    def _set_horizon(env, horizon):

        while not hasattr(env, '_max_episode_steps') and env.env != env.unwrapped:
                env = env.env

        if horizon is None:
            if not hasattr(env, '_max_episode_steps'):
                raise RuntimeError('This gym environment has no specified time limit!')
            horizon = env._max_episode_steps

        if hasattr(env, '_max_episode_steps'):
            env._max_episode_steps = np.inf  # Hack to ignore gym time limit.

        return horizon

    @staticmethod
    def _convert_gym_space(space):
        if isinstance(space, gym_spaces.Discrete):
            return Discrete(space.n)
        elif isinstance(space, gym_spaces.Box):
            return Box(low=space.low, high=space.high, shape=space.shape)
        else:
            raise ValueError

