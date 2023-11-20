from omni.isaac.kit import SimulationApp
from omni.isaac.core.world import World
from omni.isaac.core.utils.torch.maths import set_seed

from mushroom_rl.core import VectorizedEnvironment, MDPInfo
from mushroom_rl.utils.viewer import ImageViewer

# import carb


class IsaacEnv(VectorizedEnvironment):
    """
    Interface for OmniIsaacGymEnvs environments. It makes it possible to use every
    OmniIsaacGymEnvs environment just providing the task.

    """

    def __init__(self, task, sim_app_cfg_path, sim_params=None, headless=False, backend='torch'):
        """ Initializes RL and task parameters.

        Args:
            task (RLTask): The task to register to the env;
            sim_params (dict): Simulation parameters for physics settings. Defaults to None;
            headless (bool): Whether to run training headless.

        """
        RENDER_WIDTH = 1280  # 1600
        RENDER_HEIGHT = 720  # 900
        RENDER_DT = 1.0 / 60.0  # 60 Hz

        self._simulation_app = SimulationApp({"experience": sim_app_cfg_path,
                                              "headless": headless,
                                              "window_width": 1920,
                                              "window_height": 1080,
                                              "width": RENDER_WIDTH,
                                              "height": RENDER_HEIGHT})

        # TODO check if the next line is needed
        #carb.settings.get_settings().set("/persistent/omnihydra/useSceneGraphInstancing", True)

        self._sim_render = not headless

        self._viewer = ImageViewer([RENDER_WIDTH, RENDER_HEIGHT], RENDER_DT)

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
        self._world.add_task(task)
        self._task = task

        # Create MDP info for mushroom
        mdp_info = MDPInfo(self._task.observation_space, self._task.action_space, self._task._gamma,
                           self._task._max_episode_length, dt=RENDER_DT, backend=backend)
        super().__init__(mdp_info, self._task.num_envs)

    def seed(self, seed=-1):
        return set_seed(seed)

    def reset_all(self, env_mask, state=None):
        self._task.reset()
        # self._world.step(render=self._sim_render) #TODO Check if we can do otherwise
        return self._task.get_observations()

    def step_all(self, env_mask, action):
        self._task.pre_physics_step(action)

        # allow users to specify the control frequency through config
        for _ in range(self._task.control_frequency_inv):
            self._world.step(render=self._sim_render)

        observation, reward, done, info = self._task.post_physics_step()

        return observation, reward, done, info

    def render_all(self, env_mask, record=False):
        self._world.render()
        task_render = self._task.get_render()

        self._viewer.display(task_render)

        if record:
            return task_render

    def stop(self) -> None:
        self._simulation_app.close()
        return
