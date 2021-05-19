import numpy as np
import pybullet

from mushroom_rl.core import MDPInfo
from mushroom_rl.environments.pybullet import PyBullet, PyBulletObservationType
from mushroom_rl.utils.spaces import Box


class LocomotorRobot(PyBullet):
    def __init__(self, robot_path, action_spec, observation_spec, gamma, horizon,
                 debug_gui, power, joint_power=None, goal=None, c_electricity=-2.0, c_stall=-0.1, c_joints=-0.1):

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

        # Scaling terms for robot actions
        self._power = power
        self._joint_power = 100 * np.ones(len(action_spec)) if joint_power is None else joint_power

        # Superclass constructor
        super().__init__(files, action_spec, observation_spec, gamma, horizon,
                         n_intermediate_steps=4, debug_gui=debug_gui,
                         distance=3, origin=[0., 0., 0.], angles=[0., -45., 0.])

        self._client.setGravity(0, 0, -9.81)

    def setup(self, state):
        self._client.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0.0, cameraPitch=-45,
                                                cameraTargetPosition=[0., 0., 0.])

    def reward(self, state, action, next_state, absorbing):
        alive_bonus = -1 if absorbing else 1

        progress = self._compute_progress(state, next_state)

        joint_speeds = self.get_joint_velocities(next_state)
        electricity_cost = self._c_electricity * np.mean(np.abs(action * joint_speeds)) + \
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

        observation_low = np. concatenate([np.array([0, -1, -1, -3, -3, -3, -np.pi, -np.pi]), self._joint_limits_low])
        observation_high = np. concatenate([np.array([2, 1, 1, 3, 3, 3, np.pi, np.pi]), self._joint_limits_high])
        observation_space = Box(observation_low, observation_high)

        return MDPInfo(observation_space, action_space, mdp_info.gamma, mdp_info.horizon)

    def _compute_progress(self, state, next_state):
        pose_old = self.get_sim_state(state, 'torso', PyBulletObservationType.LINK_POS)
        pose_new = self.get_sim_state(next_state, 'torso', PyBulletObservationType.LINK_POS)

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

    def _compute_action(self, action):
        scaled_action = self._power * self._joint_power * np.clip(action, -1, 1)
        return scaled_action

    def _create_observation(self, state):
        pose = self.get_sim_state(state, 'torso', PyBulletObservationType.LINK_POS)
        velocity = self.get_sim_state(state, 'torso', PyBulletObservationType.LINK_LIN_VEL)

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

        body_info = [z, np.sin(angle_to_target), np.cos(angle_to_target), vx, vy, vz, roll, pitch]

        joint_pos = self.get_joint_positions(state)
        joint_vel = self.get_joint_velocities(state)

        joints = []
        for pos, vel in zip(joint_pos, joint_vel):
            joints += [pos, vel]

        return np.array(body_info + joints)



