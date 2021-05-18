import numpy as np
import pybullet
from mushroom_rl.environments.pybullet import PyBullet, PyBulletObservationType


class LocomotorRobot(PyBullet):
    def __init__(self, robot_path, action_spec, observation_spec, gamma, horizon, debug_gui,
                 c_electricity=-2.0, c_stall=-0.1, c_joints=-0.1):
        self._c_electricity = c_electricity
        self._c_stall = c_stall
        self._c_joints = c_joints

        files = {
            robot_path: dict(flags=pybullet.URDF_USE_SELF_COLLISION |
                                        pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS |
                                        pybullet.URDF_GOOGLEY_UNDEFINED_COLORS),
            'plane.urdf': {}
        }

        super().__init__(files, action_spec, observation_spec, gamma, horizon,
                         n_intermediate_steps=4, debug_gui=debug_gui,
                         distance=3, origin=[0., 0., 0.], angles=[0., -45., 0.])

        self._client.setGravity(0, 0, -9.81)
        self._potential = 0

        self._joint_pos_indexes = list()
        self._joint_speed_indexes = list()
        for joint_name, obs_type in observation_spec:
            joint_idx = self.get_sim_state_index(joint_name, obs_type)
            if obs_type == PyBulletObservationType.JOINT_VEL:
                self._joint_speed_indexes.append(joint_idx)
            elif obs_type == PyBulletObservationType.JOINT_POS:
                self._joint_pos_indexes.append(joint_idx)

    def setup(self):
        self._client.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0.0, cameraPitch=-45,
                                                cameraTargetPosition=[0., 0., 0.])

    def reward(self, state, action, next_state, absorbing):
        alive_bonus = -1 if absorbing else 1

        potential_old = self._potential
        self._potential = self._compute_potential(next_state)
        progress = self._potential - potential_old

        joint_speeds = next_state[self._joint_speed_indexes]
        electricity_cost = self._c_electricity * np.mean(np.abs(action * joint_speeds)) + \
                           self._c_stall * np.mean(np.square(action))

        joints_at_limit_cost = self._c_joints * self._count_joints_at_limit(next_state)

        return alive_bonus + progress + electricity_cost + joints_at_limit_cost

    def is_absorbing(self, state):
        raise NotImplementedError

    def _count_joints_at_limit(self, next_state):
        raise NotImplementedError

    def _compute_potential(self, next_state):
        raise NotImplementedError

