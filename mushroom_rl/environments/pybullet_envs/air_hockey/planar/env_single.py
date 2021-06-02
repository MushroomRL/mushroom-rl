import numpy as np
import pybullet_utils.transformations as transformations

from mushroom_rl.environments.pybullet_envs.air_hockey.planar.env_base import AirHockeyPlanarBase, PyBulletObservationType


class AirHockeyPlanarSingle(AirHockeyPlanarBase):
    def __init__(self, seed=None, gamma=0.99, horizon=500, timestep=1 / 240., n_intermediate_steps=1, debug_gui=False,
                 env_noise=False, obs_noise=False, obs_delay=False, control_type="torque", step_action_function=None):
        self.init_state = np.array([-0.9273, 0.9273, np.pi / 2])
        self.obs_prev = None
        super().__init__(seed=seed, gamma=gamma, horizon=horizon, timestep=timestep,
                         n_intermediate_steps=n_intermediate_steps, debug_gui=debug_gui,
                         env_noise=env_noise, n_agents=1, obs_noise=obs_noise, obs_delay=obs_delay,
                         control_type=control_type, step_action_function=step_action_function)

        self._client.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-90.0, cameraPitch=-45.0,
                                                cameraTargetPosition=[-0.5, 0., 0.])


    def _create_observation(self, state):
        puck_pose = self.get_sim_state(state, "puck", PyBulletObservationType.BODY_POS)
        puck_pose_2d = self._puck_2d_in_robot_frame(puck_pose, self.agents[0]['frame'], type='pose')

        robot_pos = np.zeros(3)
        robot_pos[0] = self.get_sim_state(state, self.agents[0]['name'] + "/joint_1", PyBulletObservationType.JOINT_POS)
        robot_pos[1] = self.get_sim_state(state, self.agents[0]['name'] + "/joint_2", PyBulletObservationType.JOINT_POS)
        robot_pos[2] = self.get_sim_state(state, self.agents[0]['name'] + "/joint_3", PyBulletObservationType.JOINT_POS)

        if self.obs_noise:
            puck_pose_2d[:2] += np.random.randn(2) * 0.001
            puck_pose_2d[2] += np.random.randn(1) * 0.001

        puck_lin_vel = self.get_sim_state(state, "puck", PyBulletObservationType.BODY_LIN_VEL)
        puck_ang_vel = self.get_sim_state(state, "puck", PyBulletObservationType.BODY_LIN_VEL)
        puck_vel_2d = self._puck_2d_in_robot_frame(np.concatenate([puck_lin_vel, puck_ang_vel]),
                                                   self.agents[0]['frame'], type='vel')
        robot_vel = np.zeros(3)
        robot_vel[0] = self.get_sim_state(state, self.agents[0]['name'] + "/joint_1",
                                          PyBulletObservationType.JOINT_VEL)
        robot_vel[1] = self.get_sim_state(state, self.agents[0]['name'] + "/joint_2",
                                          PyBulletObservationType.JOINT_VEL)
        robot_vel[2] = self.get_sim_state(state, self.agents[0]['name'] + "/joint_3",
                                          PyBulletObservationType.JOINT_VEL)

        if self.obs_delay:
            alpha = 0.5
            puck_vel_2d = alpha * puck_vel_2d + (1 - alpha) * self.obs_prev[3:6]
            robot_vel = alpha * robot_vel + (1 - alpha) * self.obs_prev[9:12]

        self.obs_prev = np.concatenate([puck_pose_2d, puck_vel_2d, robot_pos, robot_vel])
        return self.obs_prev

    def _puck_2d_in_robot_frame(self, puck_in, robot_frame, type='pose'):
        if type == 'pose':
            puck_frame = transformations.translation_matrix(puck_in[:3])
            puck_frame = puck_frame @ transformations.quaternion_matrix(puck_in[3:])

            frame_target = transformations.inverse_matrix(robot_frame) @ puck_frame
            puck_translate = transformations.translation_from_matrix(frame_target)
            _, _, puck_euler_yaw = transformations.euler_from_matrix(frame_target)

            return np.concatenate([puck_translate[:2], [puck_euler_yaw]])
        if type == 'vel':
            rot_mat = robot_frame[:3, :3]
            vec_lin = rot_mat.T @ puck_in[:3]
            return np.concatenate([vec_lin[:2], puck_in[5:6]])
