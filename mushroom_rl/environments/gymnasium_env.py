import warnings

import gymnasium as gym
from gymnasium import spaces as gym_spaces

try:
    import pybullet_envs
    pybullet_found = True
except ImportError:
    pybullet_found = False

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils.spaces import *
from mushroom_rl.utils.viewer import ImageViewer


class Gymnasium(Environment):
    """
    Interface for Gymnasium environments. It makes it possible to use every
    Gymnasium environment just providing the id, except for the Atari games that
    are managed in a separate class.

    """
    def __init__(self, name, horizon=None, gamma=0.99, headless=False, wrappers=None, wrappers_args=None, **env_args):
        """
        Constructor.

        Args:
             name (str): gymnasium id of the environment;
             horizon (int): the horizon. If None, use the one from Gymnasium;
             gamma (float, 0.99): the discount factor;
             headless (bool, False): If True, the rendering is forced to be headless.
             wrappers (list, None): list of wrappers to apply over the environment. It
                is possible to pass arguments to the wrappers by providing
                a tuple with two elements: the gym wrapper class and a
                dictionary containing the parameters needed by the wrapper
                constructor;
            wrappers_args (list, None): list of dictionaries of arguments for each wrapper;
            ** env_args: other gym environment parameters.

        """

        # MDP creation
        self._not_pybullet = True
        self._first = True
        self._headless = headless
        self._viewer = None
        if pybullet_found and '- ' + name in pybullet_envs.getList():
            import pybullet
            pybullet.connect(pybullet.DIRECT)
            self._not_pybullet = False

        self.env = gym.make(name, render_mode='rgb_array', **env_args) # always rgb_array render mode

        if wrappers is not None:
            if wrappers_args is None:
                wrappers_args = [dict()] * len(wrappers)
            for wrapper, args in zip(wrappers, wrappers_args):
                if isinstance(wrapper, tuple):
                    self.env = wrapper[0](self.env, **args, **wrapper[1])
                else:
                    self.env = wrapper(self.env, **args, **env_args)

        horizon = self._set_horizon(self.env, horizon)

        # MDP properties
        assert not isinstance(self.env.observation_space,
                              gym_spaces.MultiDiscrete)
        assert not isinstance(self.env.action_space, gym_spaces.MultiDiscrete)

        dt = self.env.unwrapped.dt if hasattr(self.env.unwrapped, "dt") else 0.1
        action_space = self._convert_gym_space(self.env.action_space)
        observation_space = self._convert_gym_space(self.env.observation_space)
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)

        if isinstance(action_space, Discrete):
            self._convert_action = lambda a: a[0]
        else:
            self._convert_action = lambda a: a

        self._seed = None

        super().__init__(mdp_info)

    def seed(self, seed):
        self._seed = seed

    def reset(self, state=None):
        if state is None:
            state, info = self.env.reset(seed=self._seed)
            self._seed = None
            return np.atleast_1d(state), info
        else:
            _, info = self.env.reset(seed=self._seed)
            self._seed = None
            self.env.state = state

            return np.atleast_1d(state), info

    def step(self, action):
        action = self._convert_action(action)
        obs, reward, absorbing, _, info = self.env.step(action) #truncated flag is ignored 

        return np.atleast_1d(obs), reward, absorbing, info

    def render(self, record=False):
        if self._first or self._not_pybullet:
            img = self.env.render()

            if self._first:
                self._viewer =  ImageViewer((img.shape[1], img.shape[0]), self.info.dt, headless=self._headless)

            self._viewer.display(img)

            self._first = False

            if record:
                return img
            else:
                return None

        return None

    def stop(self):
        try:
            if self._not_pybullet:
                self.env.close()
                
                if self._viewer is not None:
                    self._viewer.close()
        except:
            pass

    @staticmethod
    def _set_horizon(env, horizon):

        while not hasattr(env, '_max_episode_steps') and env.env != env.unwrapped:
                env = env.env

        if horizon is None:
            if not hasattr(env, '_max_episode_steps'):
                raise RuntimeError('This gymnasium environment has no specified time limit!')
            horizon = env._max_episode_steps
            if horizon == np.inf:
                warnings.warn("Horizon can not be infinity.")
                horizon = int(1e4)

        if hasattr(env, '_max_episode_steps'):
            env._max_episode_steps = horizon

        return horizon

    @staticmethod
    def _convert_gym_space(space):
        if isinstance(space, gym_spaces.Discrete):
            return Discrete(space.n)
        elif isinstance(space, gym_spaces.Box):
            return Box(low=space.low, high=space.high, shape=space.shape)
        else:
            raise ValueError

