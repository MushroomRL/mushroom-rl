import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from dm_control import suite
    from dm_control.suite.wrappers import pixels


from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import *
from mushroom_rl.utils.viewer import ImageViewer


class DMControl(Environment):
    """
    Interface for dm_control suite Mujoco environments. It makes it possible to
    use every dm_control suite Mujoco environment just providing the necessary
    information.

    """
    def __init__(self, domain_name, task_name, horizon=None, gamma=0.99, task_kwargs=None,
                 dt=.01, width_screen=480, height_screen=480, camera_id=0, 
                 use_pixels=False, pixels_width=64, pixels_height=64):
        """
        Constructor.

        Args:
             domain_name (str): name of the environment;
             task_name (str): name of the task of the environment;
             horizon (int): the horizon;
             gamma (float): the discount factor;
             task_kwargs (dict, None): parameters of the task;
             dt (float, .01): duration of a control step;
             width_screen (int, 480): width of the screen;
             height_screen (int, 480): height of the screen;
             camera_id (int, 0): position of camera to render the environment;
             use_pixels (bool, False): if True, pixel observations are used
                rather than the state vector;
             pixels_width (int, 64): width of the pixel observation;
             pixels_height (int, 464): height of the pixel observation;

        """
        # MDP creation
        self.env = suite.load(domain_name, task_name, task_kwargs=task_kwargs)
        if use_pixels:
            self.env = pixels.Wrapper(self.env, render_kwargs={'width': pixels_width, 'height': pixels_height})

        # get the default horizon
        if horizon is None:
            horizon = self.env._step_limit

        # Hack to ignore dm_control time limit.
        self.env._step_limit = np.inf

        # MDP properties
        action_space = self._convert_action_space(self.env.action_spec())
        print(self.env.observation_spec())
        observation_space = self._convert_observation_space(self.env.observation_spec())
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        self._viewer = ImageViewer((width_screen, height_screen), dt)
        self._camera_id = camera_id

        super().__init__(mdp_info)

        self._state = None

    def reset(self, state=None):
        if state is None:
            self._state = self._convert_observation(self.env.reset().observation)
        else:
            raise NotImplementedError

        return self._state

    def step(self, action):
        step = self.env.step(action)

        reward = step.reward
        self._state = self._convert_observation(step.observation)
        absorbing = step.last()

        return self._state, reward, absorbing, {}

    def render(self):
        img = self.env.physics.render(self._viewer.size[1],
                                      self._viewer.size[0],
                                      self._camera_id)
        self._viewer.display(img)

    def stop(self):
        pass

    @staticmethod
    def _convert_observation_space(observation_space):
        observation_shape = 0
        for i in observation_space:
            shape = observation_space[i].shape
            if len(shape) > 0:
                observation_shape += shape[0]
            else:
                observation_shape += 1

        return Box(low=-np.inf, high=np.inf, shape=(observation_shape,))

    @staticmethod
    def _convert_action_space(action_space):
        low = action_space.minimum
        high = action_space.maximum

        return Box(low=np.array(low), high=np.array(high))

    @staticmethod
    def _convert_observation(observation):
        obs = list()
        for i in observation:
            obs.append(np.atleast_1d(observation[i]))

        return np.concatenate(obs)
