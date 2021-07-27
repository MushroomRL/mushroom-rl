import warnings
import os

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from habitat.datasets import make_dataset
    from habitat_baselines.config.default import get_config
    from habitat_baselines.common.environments import NavRLEnv

import gym
import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.environments import Gym
from mushroom_rl.utils.spaces import Discrete, Box


class HabitatWrapper(gym.Wrapper):
    """
    This wrapper removes action 0, that by default resets the environment
    (we reset the environment only by calling env.reset()).
    It also gets only the RGB agent's view as observation, and adds the agent's
    true position in the info dictionary.

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
        obs, rwd, done, info = self.env.step(**{'action': action[0] + 1})
        obs = np.asarray(obs['rgb'])
        info.update({'position': self.get_position()})
        return obs, rwd, done, info


class HabitatNav(Gym):
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
    def __init__(self, config_file, horizon=None, gamma=0.99,
                 width=None, height=None):
        """
        Constructor.

        Args:
             config_file (str): path to the yaml file specifying the task (see
                habitat-lab/configs/tasks/ or mushroom_rl/examples/habitat_dqn);
             horizon (int, None): the horizon;
             gamma (float, 0.99): the discount factor;
             width (int, None): width of the pixel observation. If None, the
                value specified in the config file is used.
             height (int, None): height of the pixel observation. If None, the
                value specified in the config file is used.

        """
        # MDP creation
        self._not_pybullet = False
        self._first = True

        config = get_config(config_paths=config_file,
                            opts=['BASE_TASK_CONFIG_PATH', config_file])
        config.defrost()

        if horizon is None:
            horizon = config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS # Get the default horizon
        config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = horizon + 1 # Hack to ignore gym time limit

        if width is not None:
            config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = width
        if height is not None:
            config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = height

        config.freeze()

        dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE,
                               config=config.TASK_CONFIG.DATASET)
        env = NavRLEnv(config=config, dataset=dataset)
        env = HabitatWrapper(env)
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

    def reset(self, state=None):
        assert state is None, 'Cannot set Habitat state'
        return self._convert_observation(np.atleast_1d(self.env.reset()))

    def step(self, action):
        action = self._convert_action(action)
        obs, reward, absorbing, info = self.env.step(action)
        return self._convert_observation(np.atleast_1d(obs)), reward, absorbing, info

    def stop(self):
        pass

    @staticmethod
    def _convert_observation(observation):
        return observation.transpose((2, 0, 1))
