import os
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import igibson
    from igibson.envs.igibson_env import iGibsonEnv
    from igibson.utils.utils import parse_config

import gym
import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.environments import Gym
from mushroom_rl.utils.spaces import Discrete, Box
from mushroom_rl.utils.frames import LazyFrames, preprocess_frame


class iGibsonWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = env.observation_space.spaces['rgb']

    def observation(self, observation):
        return observation['rgb']


class TransposeObsWrapper(gym.ObservationWrapper):
    """
    Transposes WxHxN image observations to NxWxH.

    """
    def observation(self, observation):
        return observation.transpose((2, 0, 1))


class iGibson(Gym):
    """
    Interface for Habitat NavRLEnv with Replica scenes.
    You need to install the following repositories / datasets:
     - https://github.com/facebookresearch/habitat-lab/
     - https://github.com/facebookresearch/habitat-sim/
     - https://github.com/facebookresearch/Replica-Dataset

    The agent has to navigate from point A to point B in realistic scenes.
    Observations are pixel images of what the agent sees in front of itself.
    Image resolution is specified in the config file.
    Actions are 1 (move forward), 2 (turn left), and 3 (turn right). The amount
    of distance / degrees the agent moves / turns is specified in the config file.

    Scene details, such as the agent's initial position and orientation, are
    defined in the replica json file. If you want to try new positions, you can
    sample some from the set of the scene's navigable points, accessible by
    NavRLEnv._env._sim.sample_navigable_point().

    If you want to suppress Habitat messages run:
    export GLOG_minloglevel=2
    export MAGNUM_LOG=quiet

    """
    def __init__(self, horizon=None, gamma=0.99, is_discrete=True):
        """
        Constructor.

        Args:
             scene_name (str): name of the Replica scene where the agent is placed;
             config_path (str): path to the .yaml file specifying the task (see habitat-lab/configs/tasks/);
             horizon (int, None): the horizon;
             gamma (float, 0.99): the discount factor;

        """
        # MDP creation
        self._not_pybullet = True
        self._first = True

        config = os.path.join(igibson.root_path, 'test', 'test_house.yaml')

        env = iGibsonEnv(config_file=config, mode='headless')

        for c in env.task.termination_conditions:
            try:
                horizon = c.max_step
                break
            except:
                pass

        # env.seed(seed)
        env = iGibsonWrapper(env)
        env = TransposeObsWrapper(env)
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
                                                
        Environment.__init__(self, mdp_info)

    def stop(self):
        pass
