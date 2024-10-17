import torch
from gym import spaces as gym_spaces

from omni.isaac.kit import SimulationApp
from omniisaacgymenvs.utils.task_util import initialize_task

from mushroom_rl.core import VectorizedEnvironment, MDPInfo
from mushroom_rl.utils.viewer import ImageViewer
from mushroom_rl.utils.isaac_utils import convert_task_observation
from mushroom_rl.rl_utils.spaces import *

# import carb


class IsaacEnv(VectorizedEnvironment):
    """
    Interface for OmniIsaacGymEnvs environments. It makes it possible to use every
    OmniIsaacGymEnvs environment just providing the task.

    """

    def __init__(self, cfg=None, headless=False, backend='torch'):
        """ Initializes RL and task parameters.

        Args:
            cfg (dict): dictionary containing the parameters required to build the task;
            headless (bool): Whether to run training headless;
            backend (str, 'torch'): The backend to be used by the environment.

        """
        RENDER_WIDTH = 1280  # 1600
        RENDER_HEIGHT = 720  # 900
        RENDER_DT = 1.0 / 60.0  # 60 Hz

        self._simulation_app = SimulationApp({"headless": headless,
                                              "window_width": 1920,
                                              "window_height": 1080,
                                              "width": RENDER_WIDTH,
                                              "height": RENDER_HEIGHT})

        # TODO check if the next line is needed
        #carb.settings.get_settings().set("/persistent/omnihydra/useSceneGraphInstancing", True)

        self._render = not headless

        self._viewer = ImageViewer([RENDER_WIDTH, RENDER_HEIGHT], RENDER_DT)

        initialize_task(cfg, self)
        action_space = self._convert_gym_space(self._task.action_space)
        observation_space = self._convert_gym_space(self._task.observation_space)

        # Create MDP info for mushroom
        # default episod lenght
        max_e_lenght = 1000
        if hasattr(self._task, '_max_episode_length'):
            max_e_lenght = self._task._max_episode_length
        mdp_info = MDPInfo(observation_space, action_space, 0.99,
                           max_e_lenght, dt=RENDER_DT, backend=backend)

        super().__init__(mdp_info, self._task.num_envs)

    def set_task(self, task, backend="torch", sim_params=None, init_sim=True, rendering_dt = True, **kwargs):
        from omni.isaac.core.world import World
        RENDER_DT = 1.0 / 60.0  # 60 Hz

        self._device = "cpu"
        if sim_params and "use_gpu_pipeline" in sim_params:
            if sim_params["use_gpu_pipeline"]:
                self._device = sim_params["sim_device"]

        self._world = World(
            stage_units_in_meters=1.0,
            rendering_dt=RENDER_DT,
            backend=backend,
            sim_params=sim_params,
            device=self._device
        )

        self._task = task
        self._world.add_task(task)
        self._world.reset()

    def seed(self, seed=-1):
        from omni.isaac.core.utils.torch.maths import set_seed
        return set_seed(seed)

    def reset_all(self, env_mask, state=None):
        idxs = torch.argwhere(env_mask).squeeze()  # .cpu().numpy()  # takes torch datatype 
        if idxs.dim() > 0:  # only resets task for tensor with actual dimension
            self._task.reset_idx(idxs)
        # self._world.step(render=self._render) # TODO Check if we can do otherwise
        task_obs = self._task.get_observations()
        task_extras = self._task.get_extras()
        observation = convert_task_observation(task_obs)
        return observation.clone(), [task_extras]*self._n_envs

    def step_all(self, env_mask, action):
        self._task.pre_physics_step(action)

        # allow users to specify the control frequency through config
        for _ in range(self._task.control_frequency_inv):
            self._world.step(render=self._render)

        observation, reward, done, info = self._task.post_physics_step()
        # converts task obs from dictionary to tensor
        observation = convert_task_observation(observation)
        
        env_mask_cuda = torch.as_tensor(env_mask).cuda()
        
        return observation.clone(), reward, torch.logical_and(done, env_mask_cuda), [info]*self._n_envs

    def render_all(self, env_mask, record=False):
        self._world.render()
        task_render = self._task.get_render()

        self._viewer.display(task_render)

        if record:
            return task_render

    def stop(self):
        self._world.reset()

    def __del__(self):
        self._simulation_app.close()

    @staticmethod
    def _convert_gym_space(space):
        # import pdb; pdb.set_trace()
        if isinstance(space, gym_spaces.Discrete):
            return Discrete(space.n)
        elif isinstance(space, gym_spaces.Box):
            return Box(low=space.low, high=space.high, shape=space.shape)
        else:
            raise ValueError

    @property
    def world(self):
        return self._world

    @property
    def render_enabled(self):
        return self._render