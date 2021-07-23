import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from habitat.datasets import make_dataset
    from habitat_baselines.config.default import get_config
    from habitat_baselines.common.environments import NavRLEnv

import gym

from copy import deepcopy
from collections import deque

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.environments import Gym
from mushroom_rl.utils.spaces import Discrete, Box
from mushroom_rl.utils.frames import LazyFrames, preprocess_frame


def _get_habitat_demo():#replica_scene, replica_json, config_path):
    config = get_config(config_paths='pointnav_nomap.yaml')
    config.defrost()

    print(config.TASK_CONFIG.DATASET.SCENES_DIR)

    config.SIMULATOR.RGB_SENSOR.HFOV = 79.0
    config.SIMULATOR.RGB_SENSOR.POSITION = [0, 0.88, 0]
    config.TASK_CONFIG.DATASET.DATA_PATH = '~/habitat-baselines/ride-baselines/replica-start.json.gz'
    config.TASK_CONFIG.DATASET.SCENES_DIR += 'apartment_0'

    config.freeze()
    dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE,
                           config=config.TASK_CONFIG.DATASET)

    return config, dataset


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
    You need both habitat-lab and habitat-sim to use it.
    https://github.com/facebookresearch/habitat-lab/
    https://github.com/facebookresearch/habitat-sim/
    https://github.com/facebookresearch/Replica-Dataset


    How to use it:
     -

    Observations are egocentric views (pixel images of what the agent sees in front of itself).
    Actions are 1 (move forward), 2 (turn left), and 3 (turn right). The amount
    of distance / degrees the agent moves / turns is specified in the config file.

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

        seed = 1
        config, dataset = _get_habitat_demo()

        env = NavRLEnv(config=config, dataset=dataset)
        env = HabitatWrapper(env)
        env = TransposeObsWrapper(env)
        self.env = env

        self._img_size = env.observation_space.shape[0:2]

        # Get the default horizon
        if horizon is None:
            horizon = self.env.max_steps

        # MDP properties
        action_space = Discrete(self.env.action_space.n)
        observation_space = Box(
            low=0., high=255., shape=(3, self._img_size[1], self._img_size[0]))
        self.env.max_steps += 1 # Hack to ignore gym time limit
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        Environment.__init__(self, mdp_info)

    def step(self, action):
        obs, reward, absorbing, info = self.env.step(action)
        reward *= 1. # Int to float
        if reward > 0:
            reward = 1. # MiniGrid discounts rewards based on timesteps, but we need raw rewards

        return obs, reward, absorbing, info
