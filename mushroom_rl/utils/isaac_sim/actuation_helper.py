from enum import Enum

import torch
import warp as wp

from mushroom_rl.utils import TorchUtils


class ActuationType(Enum):
    """
    The quantity an action stands for, i.e. the mode the controlled joints are driven in.

    """
    EFFORT = "joint_efforts"
    POSITION = "joint_positions"
    VELOCITY = "joint_velocities"


class ActuationHelper:
    """
    A helper class driving the joints the agent controls.

    It owns which joints those are, applies the action to them in the mode the actuation type names, and
    reports the limits the simulator declares for them, from which the action space is built.

    """

    def __init__(self, robots, actuation_spec, actuation_type):
        """
        Constructor.

        Args:
            robots (Articulation): The articulation covering the robot of every environment.
            actuation_spec (list): The names of the joints the agent controls.
            actuation_type (ActuationType): The mode those joints are driven in.

        """
        self._robots = robots
        self._actuation_spec = actuation_spec
        self._actuation_type = actuation_type

        self._controlled_dofs = None
        self._controlled_joints = None

    def set_up(self):
        """
        Resolves the controlled joints. Isaac Sim only names the degrees of freedom of an articulation once the
        simulation is running, so this cannot happen at construction.

        """
        # kept in both forms: Isaac Sim selects joints with the warp one, torch indexing needs the other
        self._controlled_dofs = self._robots.get_dof_indices(self._actuation_spec)
        self._controlled_joints = wp.to_torch(self._controlled_dofs).long()

    def apply(self, action, env_indices=None):
        """
        Applies the given action to the controlled joints of the robot.

        Args:
            action (torch.tensor): The action to be applied to the controlled joints.
            env_indices (torch.tensor, none): The indices of the environments where
                the action should be applied. If None, the action is applied to all environments.

        """
        indices = None if env_indices is None else wp.from_torch(env_indices.to(torch.int32))
        dof_indices = self._controlled_dofs
        value = wp.from_torch(action)

        if self._actuation_type == ActuationType.EFFORT:
            self._robots.set_dof_efforts(value, indices=indices, dof_indices=dof_indices)
        elif self._actuation_type == ActuationType.POSITION:
            self._robots.set_dof_position_targets(value, indices=indices, dof_indices=dof_indices)
        else:
            self._robots.set_dof_velocity_targets(value, indices=indices, dof_indices=dof_indices)

    def get_action_limits(self):
        """
        Computes the lower and upper limits for all actions, which are the limits the simulator declares for the
        quantity the actuation type drives the joints with.

        Returns:
            Two tensors: the first contains the lower limit, and the second contains the upper limit.

        """
        if self._actuation_type == ActuationType.EFFORT:
            limit = self.get_joint_max_efforts()
            return -limit, limit
        elif self._actuation_type == ActuationType.POSITION:
            limit = self.get_joint_pos_limits()
            return limit[0], limit[1]
        else:
            limit = self.get_joint_max_velocities()
            return -limit, limit

    def get_joint_max_efforts(self):
        """
        Retrieves the maximum effort limits for the controlled joints.

        Returns:
            A tensor containing the maximum effort values for each controlled joint.

        """
        max_efforts = self._robots.get_dof_max_efforts(indices=[0], dof_indices=self._controlled_dofs)
        return wp.to_torch(max_efforts)[0].to(TorchUtils.get_device())

    def get_joint_pos_limits(self):
        """
        Retrieves the position limits for the controlled joints.

        Returns:
            A tensor containing the position limits for each controlled joint.

        """
        lower, upper = self._robots.get_dof_limits()
        dof_limits = torch.stack((wp.to_torch(lower)[0], wp.to_torch(upper)[0]), dim=1).to(TorchUtils.get_device())
        return dof_limits[self._controlled_joints].T.clone()

    def get_joint_max_velocities(self):
        """
        Retrieves the maximum velocity limits for the controlled joints.

        Returns:
            A tensor containing the maximum velocity values for each controlled joint.

        """
        max_velocities = self._robots.get_dof_max_velocities(
            indices=[0], dof_indices=self._controlled_dofs
        )
        return wp.to_torch(max_velocities)[0]

    @property
    def controlled_dofs(self):
        """
        Returns:
            The controlled joints, in the form Isaac Sim selects degrees of freedom with.

        """
        return self._controlled_dofs

    @property
    def controlled_joints(self):
        """
        Returns:
            The controlled joints, as a tensor of indices usable for torch indexing.

        """
        return self._controlled_joints
