from . import _require_launched  # noqa: F401 -- raises if Isaac Sim hasn't been launched yet

import math

import torch
import warp as wp

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.cloner import GridCloner
from isaacsim.core.experimental.materials import RigidBodyMaterial
from isaacsim.core.experimental.objects import DistantLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from pxr import PhysxSchema

from mushroom_rl.utils import TorchUtils


class SceneBuilder:
    """
    Assembles the stage an Isaac Sim environment simulates: it places one copy of the robot, clones it once per
    parallel environment, and creates the views through which the environment then reads and writes simulation
    data.

    It also owns the layout of the stage, i.e. where on it every prim lives. Everything addressing a prim derives
    from the paths this class exposes, so that the environment, its collision helper and the builder itself
    cannot disagree about where the robots are.

    """

    def __init__(self, usd_path, num_envs, env_spacing=3., collisions_between_envs=False,
                 solver_pos_it_count=None, solver_vel_it_count=None, ground_plane_friction=None):
        """
        Constructor.

        Args:
            usd_path (str): Path of the USD file describing the robot.
            num_envs (int): Number of environments to clone.
            env_spacing (float): Distance between two cloned environments.
            collisions_between_envs (bool): Whether the cloned environments collide with each other.
            solver_pos_it_count (torch.tensor, None): Position iteration count of the solver, per environment.
                It determines how accurately contacts, drives and limits are resolved; low values can improve
                performance. If None, the value stored in the USD file is kept.
            solver_vel_it_count (torch.tensor, None): Velocity iteration count of the solver, per environment.
                It determines how accurately contacts, drives and limits are resolved; low values can improve
                performance. If None, the value stored in the USD file is kept.
            ground_plane_friction (tuple, None): The static friction, dynamic friction and restitution of the
                ground plane. If None, the Isaac Sim defaults are kept.

        """
        self._usd_path = usd_path
        self._num_envs = num_envs
        self._env_spacing = env_spacing
        self._collisions_between_envs = collisions_between_envs
        self._solver_pos_it_count = solver_pos_it_count
        self._solver_vel_it_count = solver_vel_it_count
        self._ground_plane_friction = ground_plane_friction

        self._base_env_path = "/World/envs"
        self._template_env_path = self._base_env_path + "/env"
        self._zero_env_robot_path = self._template_env_path + "_0/Robot"
        self._robot_regex = self._base_env_path + "/.*/Robot"
        self._robot_glob = self._base_env_path + "/env_*/Robot"

        self._prim_paths = None
        self._robots = None
        self._physics_view = None
        self._material_buffer = None
        self._material_properties = None

    def build(self, specifications, collision_helper):
        """
        Adds the robot specified by the usd path to the stage and clones it ``num_envs`` times, then creates
        the views used to read and write data.

        Args:
            specifications (list): The observation and additional data specifications, whose paths determine
                which views have to be created.
            collision_helper (CollisionHelper): The helper tracking the contacts, prepared on the robot of
                environment 0 before it is cloned.

        Returns:
            The articulation of the robots, the views to read and write with, indexed by the path they cover,
            and the origin of each environment.

        """
        stage = stage_utils.get_current_stage()

        DistantLight("/World/defaultDistantLight").set_intensities(1000)

        # Define env_0
        stage_utils.add_reference_to_stage(self._usd_path, self._zero_env_robot_path)

        collision_helper.prepare_env(stage)

        env_pos = self._clone_envs(stage)

        robots = self._robots = Articulation(self._robot_regex, reset_xform_op_properties=False)

        pos_counts, vel_counts = self._solver_pos_it_count, self._solver_vel_it_count
        if pos_counts is not None or vel_counts is not None:
            robots.set_solver_iteration_counts(
                position_counts=None if pos_counts is None else wp.from_torch(pos_counts),
                velocity_counts=None if vel_counts is None else wp.from_torch(vel_counts)
            )

        self._create_ground_plane()

        views = self._create_views(stage, specifications, robots)

        return robots, views, env_pos

    def set_robot_friction(self, static_friction, dynamic_friction, env_indices):
        """
        Gives the collision shapes of the given environments' robots a new friction, so that a property as
        coarse as how slippery the ground is can change from episode to episode.

        The write goes through the physics view of the articulation rather than through the material prims of
        the stage: those are read once, when the simulation starts, so authoring them afterwards has no effect
        on what is simulated.

        Args:
            static_friction (torch.tensor): The static friction of each of the environments, on the host.
            dynamic_friction (torch.tensor): The dynamic friction of each of the environments, on the host.
            env_indices (torch.tensor): The environments to write, as an integer tensor.

        """
        if self._material_buffer is None:
            self._physics_view = self._robots._physics_articulation_view
            self._material_buffer = self._physics_view.get_material_properties()
            self._material_properties = wp.to_torch(self._material_buffer)

        env_indices = env_indices.to(device="cpu", dtype=torch.int32)
        self._material_properties[env_indices, :, 0] = static_friction
        self._material_properties[env_indices, :, 1] = dynamic_friction
        self._physics_view.set_material_properties(self._material_buffer, wp.from_torch(env_indices))

    @property
    def zero_env_robot_path(self):
        return self._zero_env_robot_path

    @property
    def robot_glob(self):
        return self._robot_glob

    def _clone_envs(self, stage):
        """
        Replicates the robot of environment 0 once per environment, laying the copies out on a grid.

        Args:
            stage: The stage the environments are cloned on.

        Returns:
            The origin of each environment, as a tensor.

        """
        cloner = GridCloner(spacing=self._env_spacing)
        cloner.define_base_env(self._base_env_path)
        self._prim_paths = cloner.generate_paths(self._template_env_path, self._num_envs)
        
        stage.DefinePrim(self._prim_paths[0], "Xform")

        env_pos = cloner.clone(
            source_prim_path=self._prim_paths[0],
            prim_paths=self._prim_paths,
            replicate_physics=True,
            copy_from_source=False  # Faster, but changes made to source prim will also reflect in the cloned prims
        )

        cloner.replicate_physics(
            source_prim_path=self._prim_paths[0],
            prim_paths=self._prim_paths,
            base_env_path=self._base_env_path,
            root_path=self._template_env_path + "_",
            enable_env_ids=not self._collisions_between_envs
        )

        return torch.tensor(env_pos, dtype=torch.float32, device=TorchUtils.get_device())

    def _create_views(self, stage, specifications, robots):
        """
        Creates one view per distinct path appearing in the specifications, choosing the kind of view from the
        API the prim of environment 0 carries.

        Args:
            stage: The stage the prims live on.
            specifications (list): The observation and additional data specifications.
            robots (Articulation): The articulation of the robots, serving the specifications with no path.

        Returns:
            A dictionary mapping each path to the view covering it in every environment.

        """
        views = {"": robots}

        for name, path, obs_type, element_names in specifications:
            if path not in views:
                prim = stage.GetPrimAtPath(self._zero_env_robot_path + path)
                if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                    view = Articulation(self._robot_regex + path, reset_xform_op_properties=False)
                else:
                    view = RigidPrim(self._robot_regex + path, reset_xform_op_properties=False)
                views[path] = view

        return views

    def _create_ground_plane(self):
        """
        Adds a ground plane large enough to hold every environment, with the requested friction.

        """
        size = math.ceil(self._num_envs**0.5) * self._env_spacing + 100.
        ground_plane = GroundPlane("/World/groundPlane", sizes=size)

        if self._ground_plane_friction is not None:
            static_friction, dynamic_friction, restitution = self._ground_plane_friction
            material = RigidBodyMaterial(
                "/World/Physics_Materials/groundPlane",
                static_frictions=static_friction,
                dynamic_frictions=dynamic_friction,
                restitutions=restitution
            )
            ground_plane.planes.apply_physics_materials(material)
