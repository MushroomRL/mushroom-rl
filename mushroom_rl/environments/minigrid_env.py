import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import gym_minigrid
    from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

import gym

from copy import deepcopy
from collections import deque

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.environments import Gym
from mushroom_rl.utils.spaces import Discrete, Box
from mushroom_rl.utils.frames import LazyFrames, preprocess_frame

class MiniGrid(Gym):
    """
    Interface for gym_minigrid environments. It makes it possible to
    use all MiniGrid environments that do not use text instructions, such as
    MultiRoom, KeyCorridor, BlockedUnblockPickup, ObstructedMaze.
    This environment uses either MiniGrid's default 7x7x3 observations or their
    pixel 56x56x3 version. In both cases, the state is partially observable.
    To compensate for the partial observability, LazyFrames are used.

    """
    def __init__(self, name, horizon=None, gamma=0.99, history_length=4,
                 fixed_seed=None, use_pixels=False):
        """
        Constructor.

        Args:
             name (str): name of the environment;
             horizon (int, None): the horizon;
             gamma (float, 0.99): the discount factor;
             history_length (int, 4): number of frames to form a state;
             fixed_seed (int, None): if passed, it fixes the seed of the
                environment at every reset. This way, the environment is fixed
                rather than procedurally generated;
             use_pixels (bool, False): if True, MiniGrid's default 7x7x3
                observations is converted to an image of resolution 56x56x3.

        """
        # MDP creation
        self._not_pybullet = True
        self._first = True

        env = gym.make(name)
        obs_high = 10.
        if use_pixels:
            env = RGBImgPartialObsWrapper(env) # Get pixel observations
            obs_high = 255.
        env = ImgObsWrapper(env) # Get rid of the 'mission' field
        self.env = env

        self._fixed_seed = fixed_seed

        self._img_size = env.observation_space.shape[0:2]
        self._history_length = history_length

        # Get the default horizon
        if horizon is None:
            horizon = self.env.max_steps

        # MDP properties
        action_space = Discrete(self.env.action_space.n)
        observation_space = Box(
            low=0., high=obs_high, shape=(history_length, self._img_size[1], self._img_size[0]))
        self.env.max_steps = horizon + 1 # Hack to ignore gym time limit (do not use np.inf, since MiniGrid returns r(t) = 1 - 0.9t/T)
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        Environment.__init__(self, mdp_info)

        self._state = None

    def reset(self, state=None):
        self._state = preprocess_frame(self.env.reset(), self._img_size)
        self._state = deque([deepcopy(
            self._state) for _ in range(self._history_length)],
            maxlen=self._history_length
        )
        return LazyFrames(list(self._state), self._history_length)

    def step(self, action):
        obs, reward, absorbing, info = self.env.step(action)
        reward *= 1. # Int to float
        if reward > 0:
            reward = 1. # MiniGrid discounts rewards based on timesteps, but we need raw rewards

        self._state.append(preprocess_frame(obs, self._img_size))

        return LazyFrames(list(self._state),
                          self._history_length), reward, absorbing, info
