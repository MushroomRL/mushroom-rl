import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from mj_envs.hand_manipulation_suite import *
    from mjrl.envs import *

import gym
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import *
# from mushroom_rl.utils.viewer import ImageViewer


class MuJoCoPixelObs(gym.ObservationWrapper):
    def __init__(self, env, width, height, camera_name, camera_id=0, depth=False):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=0., high=255., shape=(3, width, height))
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.depth = depth
        self.camera_id = camera_id

    def observation(self, observation):
        img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
                              camera_name=self.camera_name, camera_id=self.camera_id)
        img = img[::-1,:,:]
        return img.transpose((2, 0, 1))


class MJEnv(Environment):
    """

    """
    def __init__(self, task_name, horizon=None, gamma=0.99,
        use_pixels=False, camera_id=0, pixels_width=64, pixels_height=64):
        """
        Constructor.

        Args:
             task_name (str): name of the task;
             horizon (int): the horizon;
             gamma (float): the discount factor;
             use_pixels (bool, False): if True, pixel observations are used
                rather than the state vector;
             camera_id (int, 0): position of camera to render the environment;
             pixels_width (int, 64): width of the pixel observation;
             pixels_height (int, 64): height of the pixel observation;

        """

        self.env = gym.make(task_name)
        if use_pixels:
            self.env = MuJoCoPixelObs(self.env, width=pixels_width,
                                                height=pixels_height,
                                                camera_name='vil_camera',
                                                camera_id=camera_id)

        # # get the default horizon
        # if horizon is None:
        #     horizon = self.env._step_limit
        #
        # # Hack to ignore dm_control time limit.
        # self.env._step_limit = np.inf

        # MDP properties
        action_space = self.env.action_space
        observation_space = self.env.observation_space
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

        self._state = None

    def reset(self, state=None):
        if state is None:
            return self.env.reset()
        else:
            raise NotImplementedError

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._state = obs
        return obs, reward, done, info

    def render(self):
        pass
        # img = self.env.physics.render(self._viewer.size[1],
        #                               self._viewer.size[0],
        #                               self._camera_id)
        # self._viewer.display(img)

    def stop(self):
        pass

    # def render(self):
    #     if self._viewer is None:
    #         self._viewer = mujoco_py.MjViewer(self._sim)
    #
    #     self._viewer.render()
    #
    # def stop(self):
    #     if self._viewer is not None:
    #         v = self._viewer
    #         self._viewer = None
    #         glfw.destroy_window(v.window)
    #         del v
