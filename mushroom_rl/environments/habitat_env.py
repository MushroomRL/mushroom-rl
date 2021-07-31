import warnings
import os

with warnings.catch_warnings():

    if 'VERBOSE_HABITAT' not in os.environ: # To suppress Habitat messages
        os.environ['MAGNUM_LOG'] = 'quiet'
        os.environ['GLOG_minloglevel'] = '2'

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import habitat
    from habitat_baselines.config.default import get_config
    from habitat_baselines.common.environments import get_env_class
    from habitat_baselines.utils.env_utils import make_env_fn
    from habitat.utils.visualizations.utils import observations_to_image

import gym
import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.environments import Gym
from mushroom_rl.utils.spaces import Discrete, Box


class HabitatNavigationWrapper(gym.Wrapper):
    """
    Use it for navigation tasks, where the agent has to go from point A to B.
    Action is discrete: turn left, turn right, move forward. The amount of
    degrees / distance the agent turns / moves is defined in the YAML file.
    This wrapper also removes Habitat's default action 0, that resets the
    environment (we reset the environment only by calling env.reset()).
    The observation is the RGB agent's view of what it sees in front of itself.
    We also add the agent's true (x,y) position to the 'info' dictionary.

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


class HabitatRearrangeWrapper(gym.Wrapper):
    """
    Use it for the rearrange task, where the robot is fixed at one location and
    needs to move its arm to pick and place objects. The goal is to place all
    target objects within 15cm of their goal positions (object orientation is
    not considered).
    The observation is the RGB image returned by the sensor mounted on the head
    of the robot. We also return the end-effector position in the 'info'
    dictionary. The action is mixed.
    The first elements of the action vector are continuous values for velocity
    control of the arm's joint. The last element is a scalar value for picking /
    placing an object: if this scalar is positive and the gripper is not
    currently holding an object and the end-effector is within 15cm of an object,
    then the object closest to the end-effector is grasped; if the scalar is
    negative and the gripper is carrying an object, the object is released.

    For reward details and other task details we refer to
    https://arxiv.org/pdf/2106.14405.pdf

    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.arm_ac_size = env.action_space['ARM_ACTION']['arm_action'].shape[0]
        self.grip_ac_size = env.action_space['ARM_ACTION']['grip_action'].shape[0]
        self.n_actions = self.arm_ac_size + self.grip_ac_size
        low = np.array([0.] * self.arm_ac_size + [-1.] * self.grip_ac_size)
        high = np.ones((self.arm_ac_size + self.grip_ac_size))
        self.action_space = Box(low=low, high=high, shape=(self.n_actions,))
        self.observation_space = self.env.observation_space['robot_head_rgb']

    def reset(self):
        return np.asarray(self.env.reset()['robot_head_rgb'])

    def get_ee_position(self):
        return self.env._env._sim.robot.ee_transform.translation

    def step(self, action):
        action = {'action': 'ARM_ACTION', 'action_args':
            {'arm_action': action[:-self.grip_ac_size], 'grip_action': action[-self.grip_ac_size:]}}
        obs, rwd, done, info = self.env.step(**{'action': action})
        ee_pos = np.asarray(obs['ee_pos'])
        obs = np.asarray(obs['robot_head_rgb'])
        info.update({'ee_position': self.get_ee_position()}) # TODO: check difference
        info.update({'ee_position_x': ee_pos})
        return obs, rwd, done, info


class Habitat(Gym):
    """
    Interface for Habitat RL environments.
    This class is very generic and can be used for many Habitat task. Depending
    on the robot / task, you have to use different wrappers, since observation
    and action spaces may vary.

    See <MUSHROOM_RL PATH>/examples/habitat/ for more details.

    """
    def __init__(self, wrapper, config_file, base_config_file=None, horizon=None, gamma=0.99,
                 width=None, height=None):
        """
        Constructor. For more details on how to pass YAML configuration files,
        please see <MUSHROOM_RL PATH>/examples/habitat/README.md

        Args:
             wrapper (str): wrapper for converting observations and actions
                (e.g., HabitatRearrangeWrapper);
             config_file (str): path to the YAML file specifying the RL task
                configuration (see <HABITAT_LAB PATH>/habitat_baselines/configs/);
             base_config_file (str, None): path to an optional YAML file, used
                as 'BASE_TASK_CONFIG_PATH' in the first YAML
                (see <HABITAT_LAB PATH>/configs/);
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

        if base_config_file is None:
            base_config_file = config_file

        config = get_config(config_paths=config_file,
                opts=['BASE_TASK_CONFIG_PATH', base_config_file])

        config.defrost()

        if horizon is None:
            horizon = config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS # Get the default horizon
        config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = horizon + 1 # Hack to ignore gym time limit

        # Overwrite all RGB width / height used for the TASK (not SIMULATOR)
        for k in config['TASK_CONFIG']['SIMULATOR']:
            if 'rgb' in k.lower():
                if height is not None:
                    config['TASK_CONFIG']['SIMULATOR'][k]['HEIGHT'] = height
                if width is not None:
                    config['TASK_CONFIG']['SIMULATOR'][k]['WIDTH'] = width

        config.freeze()

        env_class = get_env_class(config.ENV_NAME)
        env = make_env_fn(env_class=env_class, config=config)
        env = globals()[wrapper](env)
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

        self._last_obs = None # For rendering

        Environment.__init__(self, mdp_info)

    def reset(self, state=None):
        assert state is None, 'Cannot set Habitat state'
        obs = self._convert_observation(np.atleast_1d(self.env.reset()))
        self._last_obs = obs
        return obs

    def step(self, action):
        action = self._convert_action(action)
        obs, reward, absorbing, info = self.env.step(action)
        self._last_obs = obs
        return self._convert_observation(np.atleast_1d(obs)), reward, absorbing, info

    def stop(self):
        pass

    def render(self, mode='rgb_array'):
        if mode == "rgb_array":
            frame = observations_to_image(
                self._last_obs, self.env.unwrapped._env.get_metrics()
            )
        else:
            raise ValueError(f"Render mode {mode} not currently supported.")

    @staticmethod
    def _convert_observation(observation):
        return observation.transpose((2, 0, 1))

    @staticmethod
    def root_path():
        return os.path.dirname(os.path.dirname(habitat.__file__))
