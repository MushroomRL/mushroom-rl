from . import _require_launched  # noqa: F401 -- raises if Isaac Sim hasn't been launched yet

import torch
import warp as wp

from functools import reduce

import isaacsim.core.experimental.utils.prim as prim_utils
from isaacsim.core.simulation_manager import SimulationManager
from pxr import PhysxSchema

from mushroom_rl.utils import TorchUtils


class CollisionHelper:
    def __init__(self, collision_groups, zero_env_robot_path, robot_glob, num_envs,
                 n_intermediate_steps=1):
        """
        Constructor.

        Args:
            collision_groups (list, None): A list containing groups of prims for which collisions should be
                checked during simulation. The entries are given as ``(key, prim_paths)``, where key is a
                string for later reference and prim_paths is a list of paths to the prims.
            zero_env_robot_path (str): Path of the robot of the first environment, which the collision groups
                are relative to.
            robot_glob (str): Wildcard matching the robot of every environment. Note that this is a wildcard
                and not a regular expression: the physics tensor views the contact views are built on match
                prim paths differently from ``isaacsim.core.experimental``.
            n_envs (int): Number of parallel environments.
            n_intermediate_steps (int): Number of intermediate control steps. Defaults to 1.

        """
        self._views = None
        self._collision_groups_indices = None
        self._collision_group_contains_world = None

        self._zero_env_robot_path = zero_env_robot_path
        self._robot_glob = robot_glob
        self._num_envs = num_envs
        self.collision_groups = {key: group for key, group in collision_groups} if collision_groups is not None else {}
        self._n_intermediate_steps = n_intermediate_steps

    def prepare_env(self, stage):
        """
        Ensures that all objects in the collision groups possess the necessary APIs.

        Args:
            stage: current stage of the isaac sim simulation
        """
        for group_name, group in self.collision_groups.items():
            for path in group:
                if path.startswith("/World/"):
                    continue
                prim = stage.GetPrimAtPath(self._zero_env_robot_path + path)
                prim_utils.ensure_api(prim, PhysxSchema.PhysxRigidBodyAPI)
                prim_utils.ensure_api(prim, PhysxSchema.PhysxContactReportAPI)

    def initialize(self):
        """
        Sets up a contact-tracking view for all objects of all collision groups.

        Has to be called once the simulation is running, since the views are built directly on the physics
        simulation view. That is one level below ``isaacsim.core.experimental.prims``, which cannot express
        these views: it resolves the body patterns to concrete prims before applying the filters, losing the
        pairing that makes each body see only the partners inside its own environment.

        Note that this layer matches paths as wildcards rather than as the regular expressions used by
        ``isaacsim.core.experimental.prims``, which is why the environment hands down a separate pattern.
        """
        simulation_view = SimulationManager.get_physics_simulation_view()

        self._views = {}
        self._collision_groups_indices = {}
        self._collision_group_contains_world = {key: False for key in self.collision_groups}

        for group_name, group in self.collision_groups.items():
            self._collision_group_contains_world[group_name] = any([path.startswith("/World/") for path in group])
            if self._collision_group_contains_world[group_name]:
                continue

            other_groups = [value for key, value in self.collision_groups.items() if key != group_name]
            possible_partners = reduce(
                lambda acc, val: acc + val if val not in acc else acc, other_groups, []
            )
            self._collision_groups_indices[group_name] = {
                key: torch.tensor([possible_partners.index(value) for value in self.collision_groups[key]],
                                  device=TorchUtils.get_device())
                for key in self.collision_groups if key != group_name
            }
            possible_partners = [self._env_pattern(partner) for partner in possible_partners]

            paths = [self._env_pattern(path) for path in group if not path.startswith("/World/")]
            self._views[group_name] = simulation_view.create_rigid_contact_view(
                paths, filter_patterns=[possible_partners] * len(paths)
            )

    def get_collision_force(self, group1, group2, selector=None, dt=1.0):
        """
        Computes the collision forces or impulses between two collision groups.

        Args:
            group1 (str): The name of the first collision group.
            group2 (str): The name of the second collision group.
            selector (Callable[[torch.tensor], torch.tensor], optional):
                A function that processes the collision force tensor.
                Defaults to selecting the maximum force of each environment
            dt (float, optional): The time step duration used for computing forces.
                The function returns contact impulses if the default dt is used

        Returns:
            A tensor containing the computed collision forces between the groups,
            processed by the `selector` function.
        """
        assert not (self._collision_group_contains_world[group1]
                    and self._collision_group_contains_world[group2])

        if selector is None:
            def selector(x):
                return torch.max(torch.max(torch.norm(x, dim=3), dim=2)[0], dim=1)[0]

        if self._collision_group_contains_world[group2]:
            group = group1
            indices_prims = self._collision_groups_indices[group1][group2]
        else:
            group = group2
            indices_prims = self._collision_groups_indices[group2][group1]
        n_bodies = len(self.collision_groups[group])

        forces = wp.to_torch(self._views[group].get_contact_force_matrix(dt))
        # transform to (num_envs, n_bodies, n_possible_partners, 3)
        forces = self._to_env_major(forces, n_bodies)
        forces = forces[:, :, indices_prims]

        return selector(forces)

    def check_collision(self, group1, group2, threshold, selector=None, dt=1.0):
        """
        Checks whether the collision force between two collision groups exceeds a given threshold.

        Args:
            group1 (str): The name of the first collision group.
            group2 (str): The name of the second collision group.
            threshold (float, torch.tensor): The threshold value to compare against.
                Can be a scalar or a tensor of the same shape as the computed forces.
            selector (Callable[[torch.tensor], torch.tensor], optional):
                A function that processes the collision force tensor.
                If None, a default selector is used. Defaults to None.
            dt (float, optional): The time step duration used for computing forces.
                The function uses impulses if the default dt is used

        Returns:
            A boolean tensor indicating where the computed forces exceed the given threshold.
        """
        forces = self.get_collision_force(group1, group2, selector, dt)
        return forces > threshold

    def count_collisions(self, group1, group2, threshold, selector=None, dt=1.0):
        """
        Counts the number of collisions between two groups, considering at most one collision between each
        possible pair of objects from the two groups.
        For example, the maximum collision count between a group with 3 objects and a group with 4 objects
        would be 12.

        Args:
            group1 (str): The name of the first collision group.
            group2 (str): The name of the second collision group.
            threshold (float, torch.tensor): The threshold value for detecting collisions.
                Can be a scalar or a tensor of the same shape as the computed forces.
            selector (Callable[[torch.tensor], torch.tensor], optional):
                A function that processes the collision force tensor.
                If None, the default selector computes the norm along the collision dimension. Defaults to None.
            dt (float, optional): The time step duration used for computing forces. Defaults to 1.0.

        Returns:
            A tensor containing the count of collisions
        """
        if selector is None:
            def selector(x):
                return torch.norm(x, dim=3)

        forces = self.get_collision_force(group1, group2, selector, dt)
        return torch.sum(torch.sum(forces > threshold, dim=2), dim=1)

    def get_net_contact_forces(self, group, dt=1.0):
        """
        Returns the net contact forces for all objects in a collision group.

        Args:
            group (str): The name of the collision group.
            dt (float, optional): The time step duration used for computing forces. Defaults to 1.0.

        Returns:
            A tensor of shape (num_envs, n_bodies, 3) with the net contact force on each body.
        """
        forces = wp.to_torch(self._views[group].get_net_contact_forces(dt))

        return self._to_env_major(forces, len(self.collision_groups[group]))

    def _env_pattern(self, path):
        """
        Turns a robot-relative prim path into a wildcard matching that prim in every environment.

        Args:
            path (str): The path of the prim, relative to the robot, or an absolute path under /World.

        Returns:
            The path as a wildcard over all environments, or unchanged if it is already absolute.

        """
        if path.startswith("/World/"):
            return path

        return self._robot_glob + path

    def _to_env_major(self, forces, n_bodies):
        """
        Reorders a contact tensor from the prim ordering of the view to an environment-major one.

        The view resolves one path pattern per body, each matching every environment, so its rows run
        body-major. The environment is the leading axis everywhere else in the environment, hence the swap.

        Args:
            forces (torch.tensor): The contact tensor, whose first axis spans ``n_bodies * num_envs`` prims.
            n_bodies (int): The number of bodies in the collision group.

        Returns:
            The same tensor with the environment as the leading axis.
        """
        assert forces.shape[0] == n_bodies * self._num_envs, \
            f"expected {n_bodies * self._num_envs} contact rows, got {forces.shape[0]}"

        return forces.view(n_bodies, self._num_envs, *forces.shape[1:]).transpose(0, 1)
