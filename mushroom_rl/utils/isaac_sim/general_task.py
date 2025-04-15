import numpy as np
import math

from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.prims import Articulation, RigidPrim, GeometryPrim
from isaacsim.core.cloner import GridCloner
from isaacsim.core.utils.types import ArticulationActions
from isaacsim.core.api.materials import PhysicsMaterial
import isaacsim.core.utils.prims as prim_utils

from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
import omni.replicator.core as rep
from omni.usd import get_context
from pxr import Gf, UsdLux, PhysxSchema

from mushroom_rl.utils.isaac_sim import ObservationType, CollisionHelper, ActionType
from mushroom_rl.core.array_backend import ArrayBackend


class GeneralTask(BaseTask):
    """
    General isaac sim taks, that can be added to the world and will handle all requests for the isaac sim environment.
    """

    BASE_ENV_PATH = "/World/envs"
    TEMPLATE_ENV_PATH = BASE_ENV_PATH + "/env"
    ZERO_ENV_PATH = TEMPLATE_ENV_PATH + "_0"

    def __init__(self, physic_context, usd_path, num_envs, env_spacing, observation_spec, actuation_spec, backend, 
                 device, action_type=ActionType.EFFORT, n_intermediate_steps=1, collision_between_envs=False, 
                 additional_data_spec=None, collision_groups=None, physics_material_spec=None, 
                 camera_position=(5, 0, 4), camera_target=(0, 0, 0), solver_pos_it_count=None, solver_vel_it_count=None,
                 ground_plane_friction=None, render_product_size=(1280, 720)):
        """
        Constructor.

        Args:
            physic_context: Physics context of the isaac sim world
            usd_path (str): Path to usd file of the robot.
            num_envs (int): Number of parallel environments.
            env_spacing (float): Distance between each environment.
            observation_spec (list): A list containing the names of data that should be made available to the agent as
               an observation and their type (ObservationType). They are combined with a path, which is used to access the prim,
               and a list or a single string with name of the subelements of prim which should be accessed. For example a subbody 
               or a joint of an Articulation
               An entry in the list is given by: (key, name, type, element). The name can later be used to retrieve
               specific observations.
            actuation_spec (list): A list specifying the names of the joints  which should be controllable by the
               agent.
            backend (str): name of the backend for array operations.
            device (str): Compute device (e.g., 'cuda:0').
            action_type (ActionType): Control type of the joints (effort, position, velocity).
            n_intermediate_steps (int): Number of intermediate control steps. Defaults to 1.
            collision_between_envs (bool): Whether inter-environment collisions are allowed.
            additional_data_spec (list, None): A list containing the data fields of interest, which should be read from
               or written to during simulation. The entries are given as the following tuples: (key, path, type) key
               is a string for later referencing in the "read_data" and "write_data" methods.
            collision_groups (list, None): A list containing groups of prims for which collisions should be checked during
                simulation. The entries are given as ``(key, prim_paths)``, where key is a string for later reference and 
                prim_paths is a list of paths to the prims.
            physics_material_spec (list, None): A list containing all data to create a custom physics material for each environment, which 
                will be applied to all rigidbodies. 
                The entries are given as the following tuples: (name, static_friction, dynamic_friction, restitution)
            camera_position (tuple): The position where the camera is placed.
            camera_target (tuple): The position the camera is aimed at.
            solver_pos_it_count (torch, array): An array with the same size as num_envs. Determines how accurately contacts, 
                drives, and limits are resolved. Low values can lead to performance improvement
            solver_vel_it_count (torch, array): An array with the same size as num_envs. Determines how accurately contacts, 
                drives, and limits are resolved. Low values can lead to performance improvement
            ground_plane_friction (tuple, None): A tuple containing the static friciton, dynamic friction and restitution 
                for the groundplane. The tuple should have the following format: (static_friction, dynamic_friction, restitution)
            render_product_size (tuple): (Width, Height) of the recorded and displayed image.
        """
        super().__init__("MushroomTask")

        self.usd_path = usd_path
        self._physic_context = physic_context
        self._num_envs = num_envs
        self._env_spacing = env_spacing
        self._collisions_between_envs = collision_between_envs
        self._observation_spec = observation_spec
        self._actuation_spec = actuation_spec
        self._additional_data_spec = additional_data_spec if additional_data_spec is not None else []
        self._backend = backend
        self._device = device
        self._action_type = action_type
        self._physics_material_spec = physics_material_spec
        self._initial_camera_pos = camera_position
        self._initial_camera_target = camera_target
        self._solver_pos_it_count = solver_pos_it_count
        self._solver_vel_it_count = solver_vel_it_count
        self._ground_plane_friction = ground_plane_friction
        self._rp_size = render_product_size

        self._consistent_property_storage = {}

        self.collision_helper = CollisionHelper(collision_groups, backend, num_envs, device, n_intermediate_steps)

    def set_up_scene(self, scene):
        """
        Called during world reset. Adds robot specified by the usd path to scene and clones ``num_envs`` 
        times. Creates various views for for reading and writing data.

        """
        super().set_up_scene(scene)
        stage = get_context().get_stage()
        self._views = {}

        #create surroundings
        self._set_camera()
        self._create_light(stage)

        #Define env_0
        prim_utils.create_prim(
            self.ZERO_ENV_PATH + "/Robot",
            usd_path=self.usd_path,
            #translation=(0., 0., 0.42)
        )

        self.collision_helper.prepare_env(stage)

        #clone env_0
        self._cloner = GridCloner(spacing=self._env_spacing)
        self._cloner.define_base_env(self.BASE_ENV_PATH)
        self.prim_paths = self._cloner.generate_paths(self.TEMPLATE_ENV_PATH, self._num_envs)
        # create source prim
        stage.DefinePrim(self.prim_paths[0], "Xform")

        self.env_pos = self._cloner.clone(
            source_prim_path=self.prim_paths[0], 
            prim_paths=self.prim_paths, 
            replicate_physics=True, 
            copy_from_source=False #Faster, but changes made to source prim will also reflect in the cloned prims
        )
        self.env_pos = np.float32(self.env_pos)
        self.env_pos = ArrayBackend.convert(self.env_pos, to=self._backend)

        self._cloner.replicate_physics(
            source_prim_path=self.prim_paths[0],
            prim_paths=self.prim_paths,
            base_env_path=self.BASE_ENV_PATH,
            root_path=self.TEMPLATE_ENV_PATH + "_",
        )
        
        #handle collisions between environments
        if not self._collisions_between_envs:
            self._cloner.filter_collisions(
                self._physic_context.prim_path,
                "/World/collisions",
                self.prim_paths,
                global_paths=["/World/groundPlane"]
            )
        
        self.robots = Articulation(
            prim_paths_expr= self.BASE_ENV_PATH + "/.*/Robot", 
            name="robot_view", 
            reset_xform_properties=False
        )
        scene.add(self.robots)

        #low iteration counts lead to performance improvements, can have sideffects
        if self._solver_pos_it_count is not None:
            self.robots.set_solver_position_iteration_counts(self._solver_pos_it_count) 
        if self._solver_vel_it_count is not None:
            self.robots.set_solver_velocity_iteration_counts(self._solver_vel_it_count)

        ground_plane_size=math.ceil(self._num_envs**0.5) * self._env_spacing + 100.
        if self._ground_plane_friction is None:
            scene.add_ground_plane(size=ground_plane_size)
        else:
            static_friction, dynamic_friction, restitution = self._ground_plane_friction
            scene.add_ground_plane(
                size=ground_plane_size, 
                static_friction=static_friction, 
                dynamic_friction=dynamic_friction, 
                restitution=restitution
            )
        
        self._views[""] = self.robots

        #register view
        self.collision_helper.set_up()

        if self._additional_data_spec is None:
            specifications = self._observation_spec 
        else:
            specifications = self._observation_spec + self._additional_data_spec
        
        for name, path, obs_type, element_names in specifications:
            if path not in self._views:
                prim = stage.GetPrimAtPath(self.ZERO_ENV_PATH + "/Robot" + path)
                if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                    view = Articulation(
                        prim_paths_expr=self.BASE_ENV_PATH + "/.*/Robot" + path,
                        name=path.replace("/", "_") + "_view",
                        reset_xform_properties=False,
                    )
                else:
                    view = RigidPrim(
                        prim_paths_expr=self.BASE_ENV_PATH + "/.*/Robot" + path,
                        name=path.replace("/", "_") + "_view",
                        reset_xform_properties=False,
                    )
                scene.add(view)
                self._views[path] = view
        
        #apply physics materials
        if self._physics_material_spec is not None:
            self._apply_physics_materials(self._physics_material_spec)

    def post_reset(self):#TODO make articulation view flexible so that it can be defined by path
        """
        Called as the last step when resetting the world.
        """
        self.collision_helper.post_reset()

        self._controlled_joints = [self.robots.get_dof_index(joint_name) for joint_name in self._actuation_spec]
        self._controlled_joints = self._arr_backend.convert(self._controlled_joints)

        self._observers = self._create_observer_tuple(self._observation_spec)
        self._additionals = self._create_observer_tuple(self._additional_data_spec)

        #reapplies specified written properties after world reset 
        for key, reapply_data in self._consistent_property_storage.items():
            reapply_data()

    def _create_observer_tuple(self, spec):
        mapping = {}
        for name, path, obs_type, element_names in spec:
            if isinstance(element_names, str):
                element_names = [element_names]

            view = self._views[path]
            element_idx = None

            # Find indices of joints or bodies
            if obs_type.is_joint() or obs_type.is_sub_body():
                index_fn = view.get_dof_index if obs_type.is_joint() else view.get_body_index
                element_idx = [index_fn(n) for n in element_names]
                element_idx = self._arr_backend.from_list(element_idx)

            mapping[name] = (view, obs_type, element_idx)
        return mapping
    
    def get_observations(self, clone=True):
        """
        Retrieves the current observations from the environment.

        Args:
            clone (bool, optional): If True, the observations are cloned to 
                avoid in-place modifications. Defaults to True.

        Returns:
            A dictionary mapping observation names to their respective values
        """
        obs = {}
        for name, (view, obs_type, element_idx) in self._observers.items():
            obs[name] = self._read_property(view, obs_type, element_idx=element_idx, clone=clone)
        return obs
    
    def apply_action(self, action, env_indices=None):
        """
        Applies the given action to the controlled joints of the robot.

        Args:
            action (torch.tensor, np.ndarray): The action to be applied to the controlled joints. 
            env_indices (torch.tensor, np.ndarray, none): The indices of the environments where 
                the action should be applied. If None, the action is applied to all environments.
        """
        kwargs = {'joint_indices': self._controlled_joints, self._action_type.value: action}
        art_action = ArticulationActions(**kwargs)
        self.robots.apply_action(art_action, indices=env_indices)

    def reset_env(self, env_indices):
        """
        Applies default values to joints and position, orientation and velocity of the robot.

        Args:
            env_indices (torch.tensor, np.ndarray, none): The indices of the environments where 
                the action should be applied. If None, the action is applied to all environments.
        """
        joint_defaults = self.robots.get_joints_default_state()
        joint_pos = joint_defaults.positions[env_indices]
        joint_vel = joint_defaults.velocities[env_indices]
        joint_eff = joint_defaults.efforts[env_indices]

        self.robots.set_joint_positions(joint_pos, indices=env_indices)
        self.robots.set_joint_velocities(joint_vel, indices=env_indices)
        self.robots.set_joint_efforts(joint_eff, indices=env_indices)

        default_state = self.robots.get_default_state()
        default_positions = default_state.positions[env_indices]
        default_orientations = default_state.orientations[env_indices]
        self.robots.set_world_poses(default_positions, default_orientations, indices=env_indices)

        velocity = self._arr_backend.zeros(len(env_indices), 6)
        self.robots.set_velocities(velocity, indices=env_indices)

    def get_observation_limits(self):
        """
        Computes the lower and upper limits for all observations.

        Returns:
            Two tensors or arrays: the first contains the lower limit, and the second contains the upper limit.
        """
        obs_low = []
        obs_high = []
        obs = self.get_observations()

        for name, (_, obs_type, joint_index) in self._observers.items():
            obs_count = self._arr_backend.size(obs[name][0, ...])

            if obs_type == ObservationType.JOINT_POS:
                limits = self.robots.get_dof_limits()
                if self._backend == "torch":
                    limits = limits.to(self._device)
                obs_low.append(limits[0, joint_index, 0])
                obs_high.append(limits[0, joint_index, 1])
            elif obs_type == ObservationType.JOINT_VEL:
                zero = self._arr_backend.zeros(1, dtype=int)
                limit = self.robots.get_joint_max_velocities(indices=zero, joint_indices=joint_index)[0]
                obs_low.append(-limit)
                obs_high.append(limit)
            else:
                inf = self._arr_backend.inf()
                obs_low.append(self._arr_backend.full((obs_count, ), -inf))
                obs_high.append(self._arr_backend.full((obs_count, ), inf))

        obs_low = self._arr_backend.concatenate(obs_low)
        obs_high = self._arr_backend.concatenate(obs_high)

        return obs_low, obs_high
    
    def get_action_limits(self):
        """
        Computes the lower and upper limits for all actions.

        Returns:
            Two tensors or arrays: the first contains the lower limit, and the second contains the upper limit.
        """
        if self._action_type == ActionType.EFFORT:
            limit = self.get_joint_max_efforts()
            return -limit, limit
        elif self._action_type == ActionType.POSITION:
            limit = self.get_joint_pos_limits()
            return limit[0], limit[1]
        else:
            limit = self.get_joint_max_velocities()
            return -limit, limit
    
    def get_joint_max_efforts(self):
        """
        Retrieves the maximum effort limits for the controlled joints.

        Returns: 
            A tensor or array containing the maximum effort values for each controlled joint.
        """
        max_efforts = self.robots.get_max_efforts(indices=[0], joint_indices=self._controlled_joints, clone=True)[0]
        if self._backend == "torch":
            max_efforts = max_efforts.to(self._device)
        return max_efforts
    
    def get_joint_pos_limits(self):
        """
        Retrieves the position limits for the controlled joints.

        Returns: 
            A tensor or array containing the position limits for each controlled joint.
        """
        dof_limits = self.robots.get_dof_limits()[0]
        if self._backend == "torch":
            dof_limits = dof_limits.to(self._device)
        return dof_limits[self._controlled_joints].T.clone()
    
    def get_joint_max_velocities(self):
        """
        Retrieves the maximum velocity limits for the controlled joints.

        Returns: 
            A tensor or array containing the maximum velocity values for each controlled joint.
        """
        return self.robots.get_joint_max_velocities(indices=[0], joint_indices=self._controlled_joints, clone=True)[0]

    def write_data(self, name, value, env_indices=None, reapply_after_reset=False):
        """
        Writes data to isaac sim.

        Args: 
            name (str): A name referring to an entry contained in additional_data_spec or observation_spec.
            value (torch.tensor, np.ndarra): The data that should be written.
            reapply_after_reset (bool): Whether the written property should be reapplied after a world reset. 
                Defaults to False.
        """
        if name in self._additionals:
            view, obs_type, element_idx = self._additionals[name]
        else:
            view, obs_type, element_idx = self._observers[name]
        self._set_property(view, obs_type, value, element_idx=element_idx, env_indices=env_indices)

        if reapply_after_reset:
            self._consistent_property_storage[name] = lambda: self._set_property(view, obs_type, value.clone(), element_idx=element_idx, env_indices=env_indices)

    def read_data(self, name, env_indices=None):
        """
        Read data from isaac sim.

        Args: 
            name (str): A name referring to an entry contained in additional_data_spec or observation_spec.

        Returns:
            The desired data as a tensor or array.
        """
        if name in self._additionals:
            view, obs_type, element_idx = self._additionals[name]
        else:
            view, obs_type, element_idx = self._observers[name]
        return self._read_property(view, obs_type, element_idx=element_idx, env_indices=env_indices, clone=False)
    
    def set_joint_data(self, value, type, element_idx=None, env_indices=None):
        """
        Sets the joint properties for the specified joints and environments.

        Args:
            value (torch.tensor, np.ndarray): The value to set for the specified joint property.
            type (ObservationType): The type of joint property to be set (e.g., "position", "velocity").
            element_idx (torch.tensor, np.ndarray, list[int], optional): The indices of the joints to update.
                If None, defaults to the controlled joints.
            env_indices (torch.tensor, np.ndarray, list[int], optional): The indices of the environments where
                the joint data should be updated. If None, applies to all environments.
        """
        if element_idx is None and type.is_joint():
            element_idx = self._controlled_joints
        self._set_property(self.robots, type, value, element_idx, env_indices)

    def teleport_away(self, env_mask):
        """
        Teleports robots away by setting their Z-coordinate to 50. This speeds up computation 
        when not all environments are used, as fewer collisions need to be processed.

        Args:
            env_indices (torch.tensor, np.ndarray, list[int]): The indices of the environments to teleport.
        """
        env_indices = self._arr_backend.where(env_mask)[0]

        pos = self.env_pos[env_indices]
        pos[:, 2] = 50
        self.robots.set_world_poses(positions=pos, indices=env_indices)
        vels = self._arr_backend.zeros(env_indices.shape[0], 6, device=self._device)
        self.robots.set_velocities(vels, indices=env_indices)

    def clear_consistent_properties(self, names=None):
        """
        Removes properties from the consistent property storage.

        Args:
            names (list[str], None): List of property names to remove.
                If None, removes all keys from the consistent property storage.
        """
        if names is None:
            self._consistent_property_storage.clear()
        else:
            for name in names:
                self._consistent_property_storage.pop(name, None)

    def _set_property(self, view, obs_type, value, element_idx=None, env_indices=None):#TODO missing max_pos_joint
        """
        Sets the specified property values immediately.

        Args:
            view: The isaac sim view where the properties should be set.
            obs_type (ObservationType): The type of observation to update.
            value (torch.tensor, np.ndarray): The new values to be assigned.
            element_idx (torch.tensor, np.ndarray, list[int], optional): The joint indices to be updated.
            env_indices (torch.tensor, np.ndarray, list[int], optional): The environment indices to apply 
                the update.
        """
        if obs_type == ObservationType.BODY_POS:
            pos = value + self.env_pos[env_indices]
            view.set_world_poses(positions=pos, indices=env_indices)
        elif obs_type == ObservationType.BODY_ROT:
            view.set_world_poses(orientations=value, indices=env_indices)
        elif obs_type == ObservationType.BODY_LIN_VEL:
            view.set_linear_velocities(value, indices=env_indices)
        elif obs_type == ObservationType.BODY_ANG_VEL:
            view.set_angular_velocities(value, indices=env_indices)
        elif obs_type == ObservationType.BODY_VEL:
            view.set_velocities(value, indices=env_indices)
        elif obs_type == ObservationType.BODY_SCALE:
            view.set_local_scales(value, indices=env_indices)
        elif obs_type == ObservationType.JOINT_POS:
            view.set_joint_positions(value, indices=env_indices, joint_indices=element_idx)
        elif obs_type == ObservationType.JOINT_VEL:
            view.set_joint_velocities(value, indices=env_indices, joint_indices=element_idx)
        elif obs_type == ObservationType.JOINT_GAIN:
            #kps is stiffness, kds is damping
            view.set_gains(kps=value[:, :, 0], kds=value[:, :, 1], indices=env_indices, joint_indices=element_idx)
        elif obs_type == ObservationType.JOINT_GAIN_STIFFNESS:
            view.set_gains(kps=value, indices=env_indices, joint_indices=element_idx)
        elif obs_type == ObservationType.JOINT_GAIN_DAMPING:
            view.set_gains(kds=value, indices=env_indices, joint_indices=element_idx)
        elif obs_type == ObservationType.JOINT_DEFAULT_POS:
            view.set_joints_default_state(positions=value, indices=env_indices, joint_indices=element_idx)
        elif obs_type == ObservationType.JOINT_MAX_EFFORT:
            view.set_max_efforts(value, indices=env_indices, joint_indices=element_idx)
        elif obs_type == ObservationType.JOINT_MAX_VELOCITY:
            view.set_max_joint_velocities(value, indices=env_indices, joint_indices=element_idx)
        elif obs_type == ObservationType.JOINT_MAX_POS:
            #can probably be implemented using usd
            raise NotImplementedError("Set function for joint max position doesn't exist in isaacsim.core.")
        elif obs_type == ObservationType.JOINT_ARMATURES:
            view.set_armatures(value, indices=env_indices, joint_indices=element_idx)
        elif obs_type == ObservationType.JOINT_FRICTION:
            view.set_friction_coefficients(value, indices=env_indices, joint_indices=element_idx)
        elif obs_type == ObservationType.JOINT_MEASURED_EFFORT:
            raise NotImplementedError("Set function for measured effort doesn't exist in isaacsim.core.")
        elif obs_type == ObservationType.SUB_BODY_INERTIA:
            view.set_body_inertias(value, indices=env_indices, body_indices=element_idx)
        elif obs_type == ObservationType.SUB_BODY_MASS:
            view.set_body_masses(value, indices=env_indices, body_indices=element_idx)
        elif obs_type == ObservationType.SUB_BODY_COM:
            view.set_body_coms(positions=value[:, :3], orientations=value[:, 3:], indices=env_indices, body_indices=element_idx)
        elif obs_type == ObservationType.SUB_BODY_COM_POS:
            view.set_body_coms(positions=value, indices=env_indices, body_indices=element_idx)
        elif obs_type == ObservationType.SUB_BODY_COM_ROT:
            view.set_body_coms(orientations=value, indices=env_indices, body_indices=element_idx)
        else:
            raise NotImplementedError()

    def _read_property(self, view, obs_type, element_idx=None, env_indices=None, clone=True):
        """
        Retrieves a specific property from the given view based on the observation type.

        Args:
            view: The view object that provides access to simulation data.
            obs_type (ObservationType): The type of observation to retrieve. 
            element_idx (torch.tensor, np.ndarray, list[int], optional): Indices of the joints 
                for which to retrieve data. Only used for joint-related observation types.
            env_indices (torch.tensor, np.ndarray, list[int], optional): Indices of the environments for 
                which to retrieve data.
            clone (bool, optional): Whether to return a cloned copy of the retrieved data. Defaults to True.

        """
        if obs_type == ObservationType.BODY_POS:
            env_pos = self.env_pos if env_indices is None else self.env_pos[env_indices]
            return view.get_world_poses(indices=env_indices, clone=clone)[0] - env_pos
        elif obs_type == ObservationType.BODY_ROT:
            return view.get_world_poses(indices=env_indices, clone=clone)[1]
        elif obs_type == ObservationType.BODY_LIN_VEL:
            return view.get_velocities(indices=env_indices, clone=clone)[:, :3]
        elif obs_type == ObservationType.BODY_ANG_VEL:
            return view.get_velocities(indices=env_indices, clone=clone)[:, 3:]
        elif obs_type == ObservationType.BODY_VEL:
            return view.get_velocities(indices=env_indices, clone=clone)
        elif obs_type == ObservationType.BODY_SCALE:
            return view.get_local_scales(indices=env_indices)
        elif obs_type == ObservationType.JOINT_POS:
            return view.get_joint_positions(indices=env_indices, joint_indices=element_idx, clone=clone)
        elif obs_type == ObservationType.JOINT_VEL:
            return view.get_joint_velocities(indices=env_indices, joint_indices=element_idx, clone=clone)
        elif obs_type == ObservationType.JOINT_GAIN:
            #kps is stiffness, kds is damping
            gains = view.get_gains(indices=env_indices, joint_indices=element_idx, clone=clone)
            return self._arr_backend.stack(gains, dim=1)
        elif obs_type == ObservationType.JOINT_GAIN_STIFFNESS:
            return view.get_gains(indices=env_indices, joint_indices=element_idx, clone=clone)[0]
        elif obs_type == ObservationType.JOINT_GAIN_DAMPING:
            return view.get_gains(indices=env_indices, joint_indices=element_idx, clone=clone)[1]
        elif obs_type == ObservationType.JOINT_DEFAULT_POS:
            view.get_joints_default_state(indices=env_indices, joint_indices=element_idx, clone=clone)#TODO
        elif obs_type == ObservationType.JOINT_MAX_EFFORT:
            return view.get_max_efforts(indices=env_indices, joint_indices=element_idx, clone=clone)
        elif obs_type == ObservationType.JOINT_MAX_VELOCITY:
            return view.get_joint_max_velocities(indices=env_indices, joint_indices=element_idx, clone=clone)
        elif obs_type == ObservationType.JOINT_MAX_POS:
            dof_limits = view.get_dof_limits()
            if self._backend == "torch":
                dof_limits = dof_limits.to(self._device)
            return dof_limits[:, self._controlled_joints]
        elif obs_type == ObservationType.JOINT_ARMATURES:
            return view.get_armatures(indices=env_indices, joint_indices=element_idx, clone=clone)
        elif obs_type == ObservationType.JOINT_FRICTION:
            return view.get_friction_coefficients(indices=env_indices, joint_indices=element_idx, clone=clone)
        elif obs_type == ObservationType.JOINT_MEASURED_EFFORT:
            return view.get_measured_joint_efforts(indices=env_indices, joint_indices=element_idx, clone=clone)
        elif obs_type == ObservationType.SUB_BODY_INERTIA:
            return view.get_body_inertias(indices=env_indices, body_indices=element_idx, clone=clone)
        elif obs_type == ObservationType.SUB_BODY_MASS:
            return view.get_body_masses(indices=env_indices, body_indices=element_idx, clone=clone)
        elif obs_type == ObservationType.SUB_BODY_COM:
            coms = view.get_body_coms(indices=env_indices, body_indices=element_idx, clone=clone)
            return self._arr_backend.concatenate(coms, dim=1)
        elif obs_type == ObservationType.SUB_BODY_COM_POS:
            return view.get_body_coms(indices=env_indices, body_indices=element_idx, clone=clone)[0]
        elif obs_type == ObservationType.SUB_BODY_COM_ROT:
            return view.get_body_coms(indices=env_indices, body_indices=element_idx, clone=clone)[1]
        else:
            raise NotImplementedError()

    def _set_camera(self):
        """
        Initializes and positions the camera in the simulation.
        """
        viewport_api = get_viewport_from_window_name("Viewport")
        viewport_api.set_active_camera("/OmniverseKit_Persp")

        self.camera_state = ViewportCameraState("/OmniverseKit_Persp", viewport_api)
        self.camera_state.set_position_world(Gf.Vec3d(self._initial_camera_pos), True)
        self.camera_state.set_target_world(Gf.Vec3d(self._initial_camera_target), True)

        rp = rep.create.render_product("/OmniverseKit_Persp", self._rp_size)
        self.rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb", do_array_copy=False, device="cuda")
        self.rgb_annot.attach(rp)

    def _create_light(self, stage, prim_path="/World/defaultDistantLight", intensity=1000):
        """Create a default light source in the scene."""
        light = UsdLux.DistantLight.Define(stage, prim_path)
        light.CreateIntensityAttr().Set(intensity)

    def _apply_physics_materials(self, values):
        """
        Creates and assigns physics materials to the robot geometries based on the provided 
        material properties.

        Args:
            values (list of tuples): A list where each entry is a tuple containing:
                - name (str): The name of the physics material.
                - static_friction (float): The static friction coefficient.
                - dynamic_friction (float): The dynamic friction coefficient.
                - restitution (float): The restitution coefficient.
        """
        materials = {}
        for i, (name, static_friction, dynamic_friction, restitution) in enumerate(values):
            
            if name not in materials:
                materials[name] = PhysicsMaterial(
                    prim_path=f"/World/Physics_Materials/{name}",
                    name=name,
                    static_friction=static_friction,
                    dynamic_friction=dynamic_friction,
                    restitution=restitution
                )
            view = GeometryPrim(self.prim_paths[i] + "/Robot", reset_xform_properties=False)
            view.apply_physics_materials(materials[name])
    
    @property
    def _arr_backend(self):
        return ArrayBackend.get_array_backend(self._backend)
