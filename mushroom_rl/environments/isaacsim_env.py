import sys

import mushroom_rl.utils.isaac_sim._require_launched  # noqa: F401 -- raises if Isaac Sim hasn't been launched yet

import torch
import warp as wp

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.simulation_manager import PhysxGpuCfg, PhysxScene, SimulationManager

from mushroom_rl.core import VectorizedEnvironment, MDPInfo
from mushroom_rl.core.spaces import Box
from mushroom_rl.utils import TorchUtils
from mushroom_rl.utils.isaac_sim import IsaacLauncher, ObservationHelper, ActuationHelper, ActuationType
from mushroom_rl.utils.isaac_sim.collision_helper import CollisionHelper
from mushroom_rl.utils.isaac_sim.scene_builder import SceneBuilder
from mushroom_rl.utils.isaac_sim.viewer import IsaacViewer


class IsaacSim(VectorizedEnvironment):
    """
    Class to create a Mushroom environment using the Isaac Sim simulator.
    """

    def __init__(self, usd_path, actuation_spec, observation_spec, num_envs, gamma, horizon,
                 timestep=None, n_substeps=1, n_intermediate_steps=1, additional_data_spec=None,
                 collision_groups=None, actuation_type=ActuationType.EFFORT, sim_params=None, scene_params=None,
                 viewer_params=None, force_fabric_update=False):
        """
        Constructor.

        Args:
            usd_path (str): Path to usd file of the robot.
            actuation_spec (list): A list specifying the names of the joints  which should be controllable by the agent.
            observation_spec (list): A list containing the names of data that should be made available to the agent as
               an observation and their type (ObservationType). They are combined with a path, which is used to access
               the prim, and a list or a single string with name of the sub elements of prim which should be accessed.
               For example a sub body or a joint of an Articulation. An entry in the list is given by:
               (key, name, type, element). The name can later be used to retrieve specific observations.
            num_envs (int): Number of parallel environments.
            gamma (float): The discounting factor of the environment.
            horizon (int): The maximum horizon for the environment.
            timestep (float, None): Simulation timestep.
            n_substeps (int, None): Number of substeps per simulation step. If None default timestep
                of isaac sim is used
            n_intermediate_steps (int): Number of intermediate control steps. Defaults to 1.
            additional_data_spec (list, None): A list containing the data fields of interest, which should be read from
               or written to during simulation. The entries are given as the following tuples: (key, path, type) key
               is a string for later referencing in the "read_data" and "write_data" methods.
            collision_groups (dict, None): A list containing groups of prims for which collisions should be checked
                during simulation. The entries are given as ``(key, prim_paths)``, where key is a string for later
                reference and prim_paths is a list of paths to the prims.
            actuation_type (ActuationType): Control type of the joints (effort, position, velocity).
            sim_params (dict): Dictionary of simulation parameters for the physics scene.
                Intended to set gpu_collision_stack_size, gpu_found_lost_aggregate_pairs_capacity,
                gpu_found_lost_pairs_capacity, gpu_heap_capacity, gpu_max_num_partitions, gpu_max_particle_contacts,
                gpu_max_rigid_contact_count, gpu_max_rigid_patch_count, gpu_max_soft_body_contacts,
                gpu_temp_buffer_capacity, gpu_total_aggregate_pairs_capacity.
            scene_params (dict, None): The parameters describing the scene to simulate, passed to
                :class:`~mushroom_rl.utils.isaac_sim.scene_builder.SceneBuilder`, which documents them.
            viewer_params (dict, None): The parameters of the camera looking at the scene, passed to
                :class:`~mushroom_rl.utils.isaac_sim.viewer.IsaacViewer`, which documents them. The ``camera_target``
                parameter defaults to env 0's position.
            force_fabric_update (bool): If True, syncs Fabric after every intermediate step regardless of
                whether rendering is disabled, instead of only before the final viewer refresh.

        """
        self._force_fabric_update = force_fabric_update

        # Isaac Sim overrides sys.stderr, which breaks tqdm — restore the original
        sys.stderr = sys.__stderr__

        self._n_intermediate_steps = n_intermediate_steps
        self._n_substeps = n_substeps

        self._num_envs = num_envs
        additional_data_spec = [] if additional_data_spec is None else additional_data_spec

        scene_params = {} if scene_params is None else scene_params
        self._scene_builder = SceneBuilder(usd_path, num_envs, **scene_params)

        self._collision_helper = CollisionHelper(collision_groups, self._scene_builder.zero_env_robot_path,
                                                 self._scene_builder.robot_glob, num_envs, n_intermediate_steps)

        self._setup_simulation(timestep, sim_params)

        specifications = observation_spec + additional_data_spec
        self._robots, views, self._env_pos = self._scene_builder.build(specifications, self._collision_helper)

        viewer_params = {k: v for k, v in ({} if viewer_params is None else viewer_params).items() if v is not None}
        env0 = tuple(self._env_pos[0].tolist())
        viewer_params.setdefault('camera_position', (env0[0] + 5, env0[1], env0[2] + 2))
        viewer_params.setdefault('camera_target', (env0[0] - 0.37, env0[1] + 2.5, env0[2] + 0.93))
        self._viewer = IsaacViewer(self.dt, **viewer_params)

        self._actuation_helper = ActuationHelper(self._robots, actuation_spec, actuation_type)
        self._observation_helper = ObservationHelper(observation_spec, additional_data_spec, num_envs, self._robots,
                                                     views, self._env_pos, self._actuation_helper)
        self._start_simulation()

        observation_limits = self._observation_helper.compute_obs_limits()
        assert observation_limits[0].dtype == observation_limits[1].dtype
        self._extend_observation_spec()
        observation_limits = self._observation_helper.obs_limits
        observation_space = Box(*observation_limits, data_type=observation_limits[0].dtype)

        action_limits = self._actuation_helper.get_action_limits()
        assert action_limits[0].dtype == action_limits[1].dtype
        action_space = Box(*action_limits, data_type=action_limits[0].dtype)

        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, self.dt, 'torch')
        mdp_info = self._modify_mdp_info(mdp_info)

        self._recompute_action_per_step = type(self)._compute_action != IsaacSim._compute_action
        self._obs = None

        super().__init__(mdp_info, num_envs)

    def step_all(self, env_mask, action):
        """
        Performs a simulation step for all active environments.

        Args:
            env_mask (torch.tensor): A boolean mask indicating which environments
                are active for this step.
            action (torch.tensor): The actions to be applied to the active environments.

        Returns:
            cur_obs (torch.tensor): The updated observations after the step.
            reward (torch.tensor): The computed rewards for each environment.
            absorbing (torch.tensor): A boolean tensor indicating if an environment is in an absorbing state.
            extra_info (dict): Additional information about the simulation step.

        """
        action = self._preprocess_action(action)

        env_indices = torch.where(env_mask)[0]
        self._teleport_away(env_mask == 0)

        ctrl_action = None

        for i in range(self._n_intermediate_steps):
            if self._recompute_action_per_step or ctrl_action is None:
                ctrl_action = self._compute_action(action)

            self._simulation_pre_step()

            self._actuation_helper.apply(ctrl_action[env_indices], env_indices)
            is_last_intermediate_step = i == self._n_intermediate_steps - 1
            update_fabric = self._force_fabric_update or \
                (not IsaacLauncher.is_rendering_disabled() and is_last_intermediate_step)
            SimulationManager.step(steps=1, update_fabric=update_fabric)

            self._simulation_post_step()

        cur_obs = self._observation_helper.observe()
        cur_obs = self._create_observation(cur_obs)

        self._step_finalize(env_indices)

        absorbing = self.is_absorbing(cur_obs)
        reward = self.reward(self._obs, action, cur_obs, absorbing)
        extra_info = self._create_info_dictionary(cur_obs)

        self._obs = cur_obs.clone()

        cur_obs = self._modify_observation(cur_obs)

        return cur_obs.clone(), reward.clone(), torch.logical_and(absorbing, env_mask).clone(), extra_info

    def reset_all(self, env_mask, state=None):
        """
        Resets the specified environments and initializes their states.

        Args:
            env_mask (torch.tensor): A boolean mask indicating which environments
                should be reset.
            state (torch.tensor): The initial state to set for the reset environments.
                Defaults to None.

        Returns:
            obs (torch.tensor): The observations after resetting the environments.
            info (dict): Additional information about the reset environments.

        """
        env_indices = torch.where(env_mask)[0]

        self._reset_env(env_indices)
        self.setup(env_indices, state)

        obs = self._observation_helper.observe()
        obs = self._create_observation(obs)
        if self._obs is None:
            self._obs = obs.clone()
        else:
            self._obs[env_mask] = obs.clone()[env_mask]

        info = self._create_info_dictionary(obs)
        obs = self._modify_observation(obs)

        return obs.clone(), info

    def render_all(self, env_mask, record=False):
        """
        Render all environments. Optionally record the frames.

        Args:
            env_mask (torch.tensor): mask selecting the environments to render.
            record (bool, False): If True, the function returns the rendered image data.

        Raises:
            RuntimeError: If ``record`` is True while rendering is disabled (see
                :meth:`~mushroom_rl.utils.isaac_sim.IsaacLauncher.activate_rendering`).

        """
        if IsaacLauncher.is_rendering_disabled():
            if record:
                raise RuntimeError("Cannot record: rendering is disabled. Call "
                                   "IsaacLauncher.activate_rendering() first.")
            return None

        data = self._viewer.render()

        return data if record else None

    def reward(self, obs, action, next_obs, absorbing):
        """
        Compute the rewards based on the given transitions.

        Args:
            obs (torch.tensor): the current states of the parallel environments.
            action (torch.tensor): the actions that are applied in the current states.
            next_obs (torch.tensor): the states reached after applying the given
                actions.
            absorbing (torch.tensor): whether next_state is an absorbing state or not.

        Returns:
            The rewards as a tensor.

        """
        raise NotImplementedError

    def is_absorbing(self, obs):
        """
        Check whether the given states are an absorbing states or not.

        Args:
            obs (torch.tensor): the states of the parallel environments.

        Returns:
            A tensor of booleans indicating whether the corresponding states are absorbing or not.

        """
        raise NotImplementedError

    def setup(self, env_indices, obs):
        """
        A function that allows to execute setup code after an environment reset.

        """
        raise NotImplementedError

    def stop(self, soft=True):
        """
        Resets simulation and closes viewer.

        If `soft` is False, the function additionally clears the consistent properties before resetting the
        simulation.

        Args:
            soft (bool): Defaults to True.
                - True: Performs soft reset of the simulation.
                - False: Perform reset of the simulation and clears consistent properties.

        """
        self._viewer.close()

        if not soft:
            self._observation_helper.clear_consistent_properties()

        self._robots.reset_to_default_state()
        self._observation_helper.reapply_consistent_properties()

    def observation_indices(self, *names):
        """
        Tells where the named observations sit in the observation vector, so that whoever consumes it can
        single them out: an asymmetric actor-critic hiding the privileged entries from the policy, a
        preprocessor normalizing one group apart from the rest, or a plot of one signal.

        Args:
            *names (str): The names of the observations to locate, as declared in the observation
                specification or added by the environment on top of it.

        Returns:
            The indices the named observations occupy, in the order the names are given.

        """
        unknown = [name for name in names if name not in self._observation_helper.obs_idx_map]
        if unknown:
            raise ValueError(f"unknown observations: {unknown}, the environment observes "
                             f"{sorted(self._observation_helper.obs_idx_map)}")

        return torch.concatenate([self._observation_helper.obs_idx_map[name] for name in names])

    @property
    def dt(self):
        return self._timestep * self._n_intermediate_steps * self._n_substeps

    @property
    def render_product_size(self):
        return self._viewer.render_product_size

    def _setup_simulation(self, timestep, custom_sim_params=None):
        """
        Create the stage and configure the physics simulation.

        Args:
            timestep (float, None): The physics timestep. If None, the default physics timestep is used.
            custom_sim_params (dict, None): A dictionary of GPU capacity parameters overriding the defaults.

        """
        stage_utils.create_new_stage()
        stage_utils.set_stage_units(meters_per_unit=1.0)

        # setting a cuda device also enables fabric, GPU dynamics and the GPU broadphase
        SimulationManager.setup_simulation(device=TorchUtils.get_device())

        self._physics_scene = SimulationManager.get_physics_scenes()[0]
        if isinstance(self._physics_scene, PhysxScene):
            self._physics_scene.set_enabled_ccd(False)
            if custom_sim_params is not None:
                self._physics_scene.set_gpu_configuration(PhysxGpuCfg(**custom_sim_params))

        if timestep is None:
            self._timestep = self._physics_scene.get_dt() / self._n_substeps
        else:
            self._timestep = timestep
        self._physics_scene.set_dt(self._timestep * self._n_substeps)

    def _start_simulation(self):
        """
        Starts the simulation and completes the setup that can only happen once physics is live.

        """
        app_utils.play(commit=True)
        app_utils.update_app()

        assert self._robots.is_physics_tensor_entity_valid(), \
            "the robot articulation did not bind to the physics simulation"

        self._capture_default_state()

        self._actuation_helper.initialize()
        self._collision_helper.initialize()
        self._observation_helper.initialize()

    def _capture_default_state(self):
        """
        Records the state the robots start in, which resets restore them to.

        Isaac Sim only reports a default state once one has been set, so the pose and joint configuration the
        USD asset was authored with have to be read back from the simulation right after it starts.

        """
        positions, orientations = self._robots.get_world_poses()
        zero_velocities = wp.from_torch(torch.zeros(self._num_envs, 3, device=TorchUtils.get_device()))
        zero_efforts = wp.from_torch(torch.zeros(self._num_envs, self._robots.num_dofs, device=TorchUtils.get_device()))

        self._robots.set_default_state(
            positions, orientations, zero_velocities, zero_velocities,
            dof_positions=self._robots.get_dof_positions(),
            dof_velocities=self._robots.get_dof_velocities(),
            dof_efforts=zero_efforts
        )

    def _reset_env(self, env_indices):
        """
        Applies default values to joints and position, orientation and velocity of the robot.

        Args:
            env_indices (torch.tensor, none): The indices of the environments where
                the action should be applied. If None, the action is applied to all environments.

        """
        indices = wp.from_torch(env_indices.to(torch.int32))
        default_state = self._robots.get_default_state(indices=indices)
        positions, orientations = default_state[0], default_state[1]
        joint_pos, joint_vel, joint_eff = default_state[4], default_state[5], default_state[6]

        self._robots.set_dof_positions(joint_pos, indices=indices)
        self._robots.set_dof_velocities(joint_vel, indices=indices)
        self._robots.set_dof_efforts(joint_eff, indices=indices)

        self._robots.set_world_poses(positions, orientations, indices=indices)

        velocity = wp.from_torch(torch.zeros(len(env_indices), 3, device=TorchUtils.get_device()))
        self._robots.set_velocities(velocity, velocity, indices=indices)

    def _teleport_away(self, env_mask):
        """
        Teleports robots away by setting their Z-coordinate to 50. This speeds up computation
        when not all environments are used, as fewer collisions need to be processed.

        Args:
            env_mask (torch.tensor): A boolean mask selecting the environments to teleport.

        """
        env_indices = torch.where(env_mask)[0]
        indices = wp.from_torch(env_indices.to(torch.int32))

        pos = self._env_pos[env_indices]
        pos[:, 2] = 50
        self._robots.set_world_poses(positions=wp.from_torch(pos), indices=indices)
        vels = wp.from_torch(torch.zeros(env_indices.shape[0], 3, device=TorchUtils.get_device()))
        self._robots.set_velocities(vels, vels, indices=indices)

    # overwritable functions ----------------------------------------------------------------------------------------

    def _create_observation(self, obs):
        """
        This method can be overridden to create a custom observation. Should be used to append observation which have
        been registered via _observation_helper.add_obs(self, name, length, min_value, max_value)

        Args:
            obs (torch.tensor): the generated observation

        Returns:
            The environment observation.

        """
        return obs

    def _modify_observation(self, obs):
        """
        This method can be overridden to edit the created observation. This is done after the reward and absorbing
        functions are evaluated. Especially useful to transform the observation into different frames. If the original
        observation order is not preserved, the helper functions in ObervationHelper breaks.

        Args:
            obs (torch.tensor): the generated observation

        Returns:
            The environment observation.

        """
        return obs

    def _compute_action(self, action):
        """
        Compute a transformation of the action at every intermediate step.
        Useful to add control signals simulated directly in python.

        Args:
            action (torch.tensor): the actions, provided at every step.

        Returns:
            The action to be applied in the isaac sim simulation

        """
        return action

    def _preprocess_action(self, action):
        """
        Compute a transformation of the action provided to the
        environment.

        Args:
            action (torch.tensor): the actions provided to the environment.

        Returns:
            The action to be used for the current step

        """
        return action

    def _extend_observation_spec(self):
        """
        This method can be overridden to register additional observations, via ``self._observation_helper
        .add_obs(...)``, before the observation space is built from them. Runs after the limits of every
        observation declared in ``observation_spec`` have been computed, so ``self._observation_helper
        .obs_idx_map`` already indexes all of them. By default, does nothing.

        """
        pass

    def _modify_mdp_info(self, mdp_info):
        """
        This method can be overridden to modify the automatically generated MDPInfo data structure.
        By default, returns the given mdp_info structure unchanged.

        Args:
            mdp_info (MDPInfo): the MDPInfo structure automatically computed by the environment.

        Returns:
            The modified MDPInfo data structure.

        """
        return mdp_info

    def _create_info_dictionary(self, obs):
        """
        This method can be overridden to create a custom info dictionary.

        Args:
            obs (torch.tensor): the generated observation

        Returns:
            The information dictionary.

        """
        return {}

    def _simulation_pre_step(self):
        """
        Allows information to be accesed and changed at every intermediate step
        before taking a step in the isaac sim simulation.
        Can be usefull to apply an external force/torque to the specified bodies.

        """
        pass

    def _simulation_post_step(self):
        """
        Allows information to be accesed at every intermediate step
        after taking a step in the isaac sim simulation.
        Can be usefull to average forces over all intermediate steps.

        """
        pass

    def _step_finalize(self, env_indices):
        """
        Allows information to be accesed at the end of a step.

        """
        pass
