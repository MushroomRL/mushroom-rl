import numpy as np
import pybullet_utils.transformations as transformations
from mushroom_rl.core import MDPInfo
from mushroom_rl.utils.spaces import Box
from mushroom_rl.environments.pybullet_envs.air_hockey.base import AirHockeyBase, PyBulletObservationType


class AirHockeyDouble(AirHockeyBase):
    """
    Base class for single agent air hockey tasks.
    """
    def __init__(self, gamma=0.99, horizon=120, env_noise=False, obs_noise=False, obs_delay=False,
                 torque_control=True, step_action_function=None, timestep=1 / 240., n_intermediate_steps=1,
                 debug_gui=False, table_boundary_terminate=False):
        self.init_state = np.array([-0.9273, 0.9273, np.pi / 2, -0.9273, 0.9273, np.pi / 2])
        self.obs_prev = None
        super().__init__(gamma=gamma, horizon=horizon, env_noise=env_noise, n_agents=2, obs_noise=obs_noise,
                         obs_delay=obs_delay, torque_control=torque_control, step_action_function=step_action_function,
                         timestep=timestep, n_intermediate_steps=n_intermediate_steps, debug_gui=debug_gui,
                         table_boundary_terminate=table_boundary_terminate)

        self._client.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-90.0, cameraPitch=-45.0,
                                                cameraTargetPosition=[-0.5, 0., 0.])
        self._change_dynamics()

    def _modify_mdp_info(self, mdp_info):
        """
        {'planar_robot_1/joint_1': {<PyBulletObservationType.JOINT_POS: 3>: [13],
                                    <PyBulletObservationType.JOINT_VEL: 4>: [16]},
         'planar_robot_1/joint_2': {<PyBulletObservationType.JOINT_POS: 3>: [14],
                                    <PyBulletObservationType.JOINT_VEL: 4>: [17]},
         'planar_robot_1/joint_3': {<PyBulletObservationType.JOINT_POS: 3>: [15],
                                    <PyBulletObservationType.JOINT_VEL: 4>: [18]},
         'planar_robot_1/link_striker_ee': {<PyBulletObservationType.LINK_POS: 5>: [19,
                                                                                    20,
                                                                                    21,
                                                                                    22,
                                                                                    23,
                                                                                    24,
                                                                                    25],
                                            <PyBulletObservationType.LINK_LIN_VEL: 6>: [26,
                                                                                        27,
                                                                                        28]},
         'planar_robot_2/joint_1': {<PyBulletObservationType.JOINT_POS: 3>: [29],
                                    <PyBulletObservationType.JOINT_VEL: 4>: [32]},
         'planar_robot_2/joint_2': {<PyBulletObservationType.JOINT_POS: 3>: [30],
                                    <PyBulletObservationType.JOINT_VEL: 4>: [33]},
         'planar_robot_2/joint_3': {<PyBulletObservationType.JOINT_POS: 3>: [31],
                                    <PyBulletObservationType.JOINT_VEL: 4>: [34]},
         'planar_robot_2/link_striker_ee': {<PyBulletObservationType.LINK_POS: 5>: [35,
                                                                                    36,
                                                                                    37,
                                                                                    38,
                                                                                    39,
                                                                                    40,
                                                                                    41],
                                            <PyBulletObservationType.LINK_LIN_VEL: 6>: [42,
                                                                                        43,
                                                                                        44]},
         'puck': {<PyBulletObservationType.BODY_POS: 0>: [0, 1, 2, 3, 4, 5, 6],
                  <PyBulletObservationType.BODY_LIN_VEL: 1>: [7, 8, 9],
                  <PyBulletObservationType.BODY_ANG_VEL: 2>: [10, 11, 12]}}
        """
        obs_idx = [0, 1, 2, 7, 8, 9, 13, 14, 15, 16, 17, 18, 0, 1, 2, 7, 8, 9, 29, 30, 31, 32, 33, 34]
        obs_low = mdp_info.observation_space.low[obs_idx]
        obs_high = mdp_info.observation_space.high[obs_idx]
        obs_low[0:3] = [-1, -0.5, -np.pi]
        obs_high[0:3] = [1, 0.5, np.pi]
        obs_low[12:15] = [-1, -0.5, -np.pi]
        obs_high[12:15] = [1, 0.5, np.pi]
        observation_space = Box(low=obs_low, high=obs_high)

        return MDPInfo(observation_space, mdp_info.action_space, mdp_info.gamma, mdp_info.horizon)


    def _create_observation(self, state):
        puck_pose = self.get_sim_state(state, "puck", PyBulletObservationType.BODY_POS)
        obs = np.zeros(24)
        for i in range(2):
            puck_pose_2d = self._puck_2d_in_robot_frame(puck_pose, self.agents[i]['frame'], type='pose')

            robot_pos = np.zeros(3)
            robot_pos[0] = self.get_sim_state(state, self.agents[i]['name'] + "/joint_1", PyBulletObservationType.JOINT_POS)
            robot_pos[1] = self.get_sim_state(state, self.agents[i]['name'] + "/joint_2", PyBulletObservationType.JOINT_POS)
            robot_pos[2] = self.get_sim_state(state, self.agents[i]['name'] + "/joint_3", PyBulletObservationType.JOINT_POS)

            if self.obs_noise:
                puck_pose_2d[:2] += np.random.randn(2) * 0.001
                puck_pose_2d[2] += np.random.randn(1) * 0.001

            puck_lin_vel = self.get_sim_state(state, "puck", PyBulletObservationType.BODY_LIN_VEL)
            puck_ang_vel = self.get_sim_state(state, "puck", PyBulletObservationType.BODY_ANG_VEL)
            puck_vel_2d = self._puck_2d_in_robot_frame(np.concatenate([puck_lin_vel, puck_ang_vel]),
                                                       self.agents[i]['frame'], type='vel')
            robot_vel = np.zeros(3)
            robot_vel[0] = self.get_sim_state(state, self.agents[i]['name'] + "/joint_1",
                                              PyBulletObservationType.JOINT_VEL)
            robot_vel[1] = self.get_sim_state(state, self.agents[i]['name'] + "/joint_2",
                                              PyBulletObservationType.JOINT_VEL)
            robot_vel[2] = self.get_sim_state(state, self.agents[i]['name'] + "/joint_3",
                                              PyBulletObservationType.JOINT_VEL)

            if self.obs_delay:
                alpha = 0.5
                puck_vel_2d = alpha * puck_vel_2d + (1 - alpha) * self.obs_prev[i][3:6]
                robot_vel = alpha * robot_vel + (1 - alpha) * self.obs_prev[i][9:12]

            obs[i * 12:(i+1) * 12] = np.concatenate([puck_pose_2d, puck_vel_2d, robot_pos, robot_vel])
        self.obs_prev = obs
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

    def _change_dynamics(self):
        for i in range(5):
            self.client.changeDynamics(self._model_map['planar_robot_1'], i, linearDamping=0., angularDamping=0.)
            self.client.changeDynamics(self._model_map['planar_robot_2'], i, linearDamping=0., angularDamping=0.)
