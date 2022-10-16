import numpy as np
import pybullet_utils.transformations as transformations
from mushroom_rl.core import MDPInfo
from mushroom_rl.utils.spaces import Box
from mushroom_rl.environments.pybullet_envs.air_hockey.base import AirHockeyBaseBullet, PyBulletObservationType


class AirHockeySingleBullet(AirHockeyBaseBullet):
    """
    Base class for single agent air hockey tasks.
    """
    def __init__(self, gamma=0.99, horizon=120, env_noise=False, obs_noise=False, obs_delay=False,
                 torque_control=True, step_action_function=None, timestep=1 / 240., n_intermediate_steps=1,
                 debug_gui=False, table_boundary_terminate=False, number_flags=0):

        """
        Constructor.

        Args:
            number_flags(int, 0): Amount of flags which are added to the observation space
        """
        self.init_state = np.array([-0.9273, 0.9273, np.pi / 2])
        self.number_flags = number_flags
        self.obs_prev = None
        super().__init__(gamma=gamma, horizon=horizon, env_noise=env_noise, n_agents=1, obs_noise=obs_noise,
                         obs_delay=obs_delay, torque_control=torque_control, step_action_function=step_action_function,
                         timestep=timestep, n_intermediate_steps=n_intermediate_steps, debug_gui=debug_gui,
                         table_boundary_terminate=table_boundary_terminate)

        self._client.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-90.0, cameraPitch=-45.0,
                                                cameraTargetPosition=[-0.5, 0., 0.])
        self._change_dynamics()
        self._disable_collision()

    def _modify_mdp_info(self, mdp_info):
        """
        puck position indexes: [0, 1]
        puck velocity indexes: [7, 8, 9]
        joint position indexes: [13, 14, 15]
        joint velocity indexes: [16, 17, 18]
        """
        obs_idx = [0, 1, 7, 8, 9, 13, 14, 15, 16, 17, 18]
        obs_low = np.append(mdp_info.observation_space.low[obs_idx], [0] * self.number_flags)
        obs_high = np.append(mdp_info.observation_space.high[obs_idx], [1] * self.number_flags)
        obs_low[0:2] = [-1, -0.5]
        obs_high[0:2] = [1, 0.5]
        observation_space = Box(low=obs_low, high=obs_high)
        return MDPInfo(observation_space, mdp_info.action_space, mdp_info.gamma, mdp_info.horizon)

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
        puck_ang_vel = self.get_sim_state(state, "puck", PyBulletObservationType.BODY_ANG_VEL)
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
            puck_vel_2d = alpha * puck_vel_2d + (1 - alpha) * self.obs_prev[2:6]
            robot_vel = alpha * robot_vel + (1 - alpha) * self.obs_prev[8:11]

        self.obs_prev = np.concatenate([puck_pose_2d, puck_vel_2d, robot_pos, robot_vel])
        return self.obs_prev

    def _puck_2d_in_robot_frame(self, puck_in, robot_frame, type='pose'):
        if type == 'pose':
            puck_frame = transformations.translation_matrix(puck_in[:3])
            puck_frame = puck_frame @ transformations.quaternion_matrix(puck_in[3:])

            frame_target = transformations.inverse_matrix(robot_frame) @ puck_frame
            puck_translate = transformations.translation_from_matrix(frame_target)

            return puck_translate[:2]
        if type == 'vel':
            rot_mat = robot_frame[:3, :3]
            vec_lin = rot_mat.T @ puck_in[:3]
            return np.concatenate([vec_lin[:2], puck_in[5:6]])

    def _change_dynamics(self):
        for i in range(5):
            self.client.changeDynamics(self._model_map['planar_robot_1'], i, linearDamping=0., angularDamping=0.)

    def _disable_collision(self):
        iiwa_links = ['planar_robot_1/link_1', 'planar_robot_1/link_2', 'planar_robot_1/link_3',
                      'planar_robot_1/link_striker_hand', 'planar_robot_1/link_striker_ee']
        table_rims = ['t_down_rim_l', 't_down_rim_r', 't_up_rim_r', 't_up_rim_l',
                      't_left_rim', 't_right_rim', 't_base', 't_up_rim_top', 't_down_rim_top', 't_base']
        for iiwa_l in iiwa_links:
            for table_r in table_rims:
                self.client.setCollisionFilterPair(self._indexer.link_map[iiwa_l][0],
                                                   self._indexer.link_map[table_r][0],
                                                   self._indexer.link_map[iiwa_l][1],
                                                   self._indexer.link_map[table_r][1], 0)

        self.client.setCollisionFilterPair(self._model_map['puck'], self._indexer.link_map['t_down_rim_top'][0],
                                           -1, self._indexer.link_map['t_down_rim_top'][1], 0)
        self.client.setCollisionFilterPair(self._model_map['puck'], self._indexer.link_map['t_up_rim_top'][0],
                                           -1, self._indexer.link_map['t_up_rim_top'][1], 0)