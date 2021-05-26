import numpy as np
import pybullet

from mushroom_rl.core import MDPInfo
from mushroom_rl.environments.pybullet import PyBullet, PyBulletObservationType
from mushroom_rl.utils.spaces import Box

from itertools import product


class LocomotorRobot(PyBullet):
    def __init__(self, robot_path, joints, gamma, horizon, debug_gui, power, joint_power, robot_name=None, goal=None,
                 c_electricity=-2.0, c_stall=-0.1, c_joints=-0.1):

        self._robot_name = robot_name
        self._goal = np.array([1e3, 0]) if goal is None else goal
        self._c_electricity = c_electricity
        self._c_stall = c_stall
        self._c_joints = c_joints

        files = {
            robot_path: dict(flags=pybullet.URDF_USE_SELF_COLLISION |
                                   pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS |
                                   pybullet.URDF_GOOGLEY_UNDEFINED_COLORS),
            'plane.urdf': {}
        }

        # Build observation and action spec
        action_spec = [(j, pybullet.TORQUE_CONTROL) for j in joints]

        observation_types = [PyBulletObservationType.JOINT_POS, PyBulletObservationType.JOINT_VEL]
        observation_spec = [obs for obs in product(joints, observation_types)]

        if self._robot_name:
            observation_spec += [
                (robot_name, PyBulletObservationType.BODY_POS),
                (robot_name, PyBulletObservationType.BODY_LIN_VEL)
            ]
        else:
            observation_spec += [
                ("torso", PyBulletObservationType.LINK_POS),
                ("torso", PyBulletObservationType.LINK_LIN_VEL)
            ]

        # Scaling terms for robot actions
        self._power = power
        self._joint_power = joint_power

        # Superclass constructor
        super().__init__(files, action_spec, observation_spec, gamma, horizon,
                         n_intermediate_steps=4, debug_gui=debug_gui,
                         distance=3, origin=[0., 0., 0.], angles=[0., -45., 0.])

        self._client.setGravity(0, 0, -9.81)

    def setup(self, state):
        self._client.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0.0, cameraPitch=-45,
                                                cameraTargetPosition=[0., 0., 0.])

        model_id = 0
        for j in range(self._client.getNumJoints(model_id)):
            joint_pos = np.random.uniform(low=-0.1, high=0.1)

            joint_info = self._client.getJointInfo(model_id, j)
            low = joint_info[8]
            high = joint_info[9]
            joint_pos = np.clip(joint_pos, low, high)

            joint_vel = 0

            self._client.resetJointState(0, j, targetValue=joint_pos, targetVelocity=joint_vel)

    def reward(self, state, action, next_state, absorbing):
        alive_bonus = -1 if absorbing else 1

        progress = self._compute_progress(state, next_state)

        joint_speeds = self.get_joint_velocities(next_state)
        electricity_cost = self._c_electricity * np.mean(np.abs(action * 0.1 * joint_speeds)) + \
                           self._c_stall * np.mean(np.square(action))

        joints_at_limit_cost = self._c_joints * self._count_joints_at_limit(next_state)

        return alive_bonus + progress + electricity_cost + joints_at_limit_cost

    def is_absorbing(self, state):
        raise NotImplementedError

    def _modify_mdp_info(self, mdp_info):
        n_actions = mdp_info.action_space.shape[0]

        action_low = -np.ones(n_actions)
        action_high = np.ones(n_actions)
        action_space = Box(action_low, action_high)

        joints_low, joints_high = self.get_joint_limits()
        velocity_limits = 10*np.ones(joints_low.shape[0])

        observation_low = np.concatenate([np.array([0, -1, -1, -3, -3, -3, -np.pi, -np.pi]),
                                          joints_low, -velocity_limits])
        observation_high = np.concatenate([np.array([2, 1, 1, 3, 3, 3, np.pi, np.pi]),
                                           joints_high, velocity_limits])

        observation_space = Box(observation_low, observation_high)

        return MDPInfo(observation_space, action_space, mdp_info.gamma, mdp_info.horizon)

    def _compute_progress(self, state, next_state):
        pose_old = self._get_torso_pos(state)
        pose_new = self._get_torso_pos(next_state)

        old_distance = np.linalg.norm(pose_old[:2] - self._goal)
        new_distance = np.linalg.norm(pose_new[:2] - self._goal)

        old_potential = -old_distance / self.dt
        new_potential = -new_distance / self.dt

        return new_potential - old_potential

    def _count_joints_at_limit(self, next_state):
        pos_joints = self.get_joint_positions(next_state)

        low, high = self.get_joint_limits()

        low_saturations = np.sum(pos_joints < low*0.99)
        high_saturations = np.sum(pos_joints > high*0.99)

        return float(low_saturations + high_saturations)

    def _compute_action(self, state, action):
        scaled_action = self._power * self._joint_power * np.clip(action, -1, 1)
        return scaled_action

    def _get_torso_pos(self, state):
        if self._robot_name:
            return self.get_sim_state(state, self._robot_name, PyBulletObservationType.BODY_POS)
        else:
            return self.get_sim_state(state, 'torso', PyBulletObservationType.LINK_POS)

    def _get_torso_vel(self, state):
        if self._robot_name:
            return self.get_sim_state(state, self._robot_name, PyBulletObservationType.BODY_LIN_VEL)
        else:
            return self.get_sim_state(state, 'torso', PyBulletObservationType.LINK_LIN_VEL)

    def _create_observation(self, state):
        pose = self._get_torso_pos(state)
        velocity = self._get_torso_vel(state)

        euler = pybullet.getEulerFromQuaternion(pose[3:])
        z = pose[2]
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]

        goal_error = self._goal - pose[:2]
        goal_angle = np.arctan2(goal_error[1], goal_error[0])
        angle_to_target = goal_angle - yaw

        rot_speed = np.array([[np.cos(-yaw), -np.sin(-yaw), 0],
                              [np.sin(-yaw),  np.cos(-yaw), 0],
                              [0, 0, 1]])
        vx, vy, vz = np.dot(rot_speed, velocity)  # rotate speed back to body point of view

        body_info = np.array([z, np.sin(angle_to_target), np.cos(angle_to_target), vx, vy, vz, roll, pitch])

        joint_pos = self.get_joint_positions(state)
        joint_vel = self.get_joint_velocities(state)

        return np.concatenate([body_info, joint_pos, joint_vel])
