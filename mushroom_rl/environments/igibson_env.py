import os
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    import logging
    logging.disable(logging.CRITICAL + 1) # Disable iGibson import log messages

    import igibson
    from igibson.envs.igibson_env import iGibsonEnv
    from igibson.utils.utils import parse_config

    logging.disable(logging.NOTSET) # Re-enable logging

import gym
import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.environments import Gym
from mushroom_rl.utils.spaces import Discrete, Box
from mushroom_rl.utils.viewer import ImageViewer

class iGibsonWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = env.observation_space.spaces['rgb']

    def observation(self, observation):
        return observation['rgb'] * 255.


class iGibson(Gym):
    """
    Interface for iGibson https://github.com/StanfordVL/iGibson

    There are both navigation and interaction tasks.
    Observations are pixel images of what the agent sees in front of itself.
    Image resolution is specified in the config file.
    By default, actions are continuous, but can be discretized automatically
    using a flag. Note that not all robots support discrete actions.

    Scene and task details are defined in the YAML config file.

    """
    def __init__(self, config_file, horizon=None, gamma=0.99, is_discrete=False,
                 width=None, height=None, debug_gui=False, verbose=False):
        """
        Constructor.

        Args:
             config_file (str): path to the YAML file specifying the task
                (see igibson/examples/configs/ and igibson/test/);
             horizon (int, None): the horizon;
             gamma (float, 0.99): the discount factor;
             is_discrete (bool, False): if True, actions are automatically
                discretized by iGibson's `set_up_discrete_action_space`.
                Please note that not all robots support discrete actions.
             width (int, None): width of the pixel observation. If None, the
                value specified in the config file is used;
             height (int, None): height of the pixel observation. If None, the
                value specified in the config file is used;
             debug_gui (bool, False): if True, activate the iGibson in GUI mode,
                showing the pybullet rendering and the robot camera.
             verbose (bool, False): if False, it disable iGibson default messages.

        """

        if not verbose:
            logging.disable(logging.CRITICAL + 1) # Disable iGibson log messages

        # MDP creation
        self._not_pybullet = False
        self._first = True

        config = parse_config(config_file)
        config['is_discrete'] = is_discrete

        if horizon is not None:
            config['max_step'] = horizon
        else:
            horizon = config['max_step']
            config['max_step'] = horizon + 1 # Hack to ignore gym time limit

        if width is not None:
            config['image_width'] = width
        if height is not None:
            config['image_height'] = height

        env = iGibsonEnv(config_file=config, mode='gui' if debug_gui else 'headless')
        env = iGibsonWrapper(env)

        self.env = env

        self._img_size = env.observation_space.shape[0:2]

        # MDP properties
        action_space = self.env.action_space
        observation_space = Box(
            low=0., high=255., shape=(3, self._img_size[1], self._img_size[0]))
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        if isinstance(action_space, Discrete):
            self._convert_action = lambda a: a[0]
        else:
            self._convert_action = lambda a: a

        self._viewer = ImageViewer((self._img_size[1], self._img_size[0]), 1/60)
        self._image = None

        Environment.__init__(self, mdp_info)

    def reset(self, state=None):
        assert state is None, 'Cannot set iGibson state'
        return self._convert_observation(np.atleast_1d(self.env.reset()))

    def step(self, action):
        action = self._convert_action(action)
        obs, reward, absorbing, info = self.env.step(action)
        self._image = obs.copy()
        return self._convert_observation(np.atleast_1d(obs)), reward, absorbing, info

    def close(self):
        self.env.close()

    def stop(self):
        self._viewer.close()

    def render(self, mode='human'):
        self._viewer.display(self._image)

    @staticmethod
    def _convert_observation(observation):
        return observation.transpose((2, 0, 1))

    @staticmethod
    def root_path():
        return igibson.root_path
