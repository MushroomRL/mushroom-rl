import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from habitat.datasets import make_dataset
    from habitat_baselines.config.default import get_config
    from habitat_baselines.common.environments import NavRLEnv

import gym
import numpy as np

from copy import deepcopy
from collections import deque

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.environments import Gym
from mushroom_rl.utils.spaces import Discrete, Box
from mushroom_rl.utils.frames import LazyFrames, preprocess_frame


class HabitatWrapper(gym.Wrapper):
    """
    - By default, action 0 resets the environment, so we do not want to use it
    (we reset the environment by calling env.reset()).
    - We take only RGB view as observation.
    - We add agent's true position in the info dictionary.

    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.action_space = gym.spaces.Discrete(env.action_space.n - 1)
        self.observation_space = self.env.observation_space['rgb']

    def reset(self):
        return np.asarray(self.env.reset()['rgb'])
    
    def get_position(self):
        return self.env._env._sim.get_agent_state().position

    def step(self, action):
        obs, rwd, done, info = self.env.step(**{'action': action + 1})
        obs = np.asarray(obs['rgb'])
        info.update({'position': self.get_position()})
        return obs, rwd, done, info


class TransposeObsWrapper(gym.ObservationWrapper):
    """
    Transposes WxHxN image observations to NxWxH.

    """
    def observation(self, observation):
        return observation.transpose((2, 0, 1))


class HabitatNavRL(Gym):
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
    def __init__(self, horizon=None, gamma=0.99):
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

        config = get_config(config_paths='pointnav_nomap.yaml')
        config.defrost()

        if horizon is None:
            horizon = config.ENVIRONMENT.MAX_EPISODE_STEPS # Get the default horizon
        config.ENVIRONMENT.MAX_EPISODE_STEPS = horizon + 1 # Hack to ignore gym time limit

        config.SIMULATOR.RGB_SENSOR.HFOV = 79.0
        config.SIMULATOR.RGB_SENSOR.POSITION = [0, 0.88, 0]
        config.TASK_CONFIG.DATASET.DATA_PATH = 'replica-start.json.gz'
        config.TASK_CONFIG.DATASET.SCENES_DIR += 'apartment_0'

        config.freeze()
        dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE,
                               config=config.TASK_CONFIG.DATASET)

        env = NavRLEnv(config=config, dataset=dataset)
        env = HabitatWrapper(env)
        env = TransposeObsWrapper(env)
        self.env = env

        self._img_size = env.observation_space.shape[0:2]

        # MDP properties
        action_space = Discrete(self.env.action_space.n)
        observation_space = Box(
            low=0., high=255., shape=(3, self._img_size[1], self._img_size[0]))
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        self._convert_action = lambda a: a[0]

        Environment.__init__(self, mdp_info)

    def stop(self):
        pass

