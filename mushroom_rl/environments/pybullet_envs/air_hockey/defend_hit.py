import time
import math
import pybullet as p
from matplotlib import image
from mushroom_rl.core import MDPInfo
from mushroom_rl.utils.spaces import Box

import numpy as np
from mushroom_rl.environments.pybullet_envs.air_hockey.double import AirHockeyDouble, PyBulletObservationType


class AirHockeyDefendHit(AirHockeyDouble):
    """
        Class for the air hockey hitting task.
        The agent tries to get close to the puck if the hitting does not happen.
        And will get bonus reward if the robot scores a goal.
        """

    def __init__(self, gamma=0.99, horizon=120, env_noise=False, obs_noise=False, obs_delay=False, torque_control=True,
                 step_action_function=None, timestep=1 / 240., n_intermediate_steps=1, debug_gui=False,
                 random_init=False, action_penalty=1e-3, table_boundary_terminate=False, hit_agent=None):
        """
        Constructor

        Args:
            random_init(bool, False): If true, initialize the puck at random position.
            action_penalty(float, 1e-3): The penalty of the action on the reward at each time step
        """
        self.hit_range = np.array([[0.25, 0.65], [-0.4, 0.4]])

        self.random_init = random_init
        self.action_penalty = action_penalty
        self.hit_agent = hit_agent
        super().__init__(gamma=gamma, horizon=horizon, timestep=timestep, n_intermediate_steps=n_intermediate_steps,
                         debug_gui=debug_gui, env_noise=env_noise, obs_noise=obs_noise, obs_delay=obs_delay,
                         torque_control=torque_control, step_action_function=step_action_function,
                         table_boundary_terminate=table_boundary_terminate)

        self.has_defend = False
        self.has_hit = False
        self.has_bounce = False


    def setup(self, state):
        if self.random_init:
            self.puck_pos = np.random.rand(2) * (self.hit_range[:, 1] - self.hit_range[:, 0]) + self.hit_range[:, 0]
        else:
            self.puck_pos = np.mean(self.hit_range, axis=1)

        puck_pos = np.concatenate([self.puck_pos, [-0.189]])
        # Change Quaternion from [0, 0, 0, 1] to [0, 0, 1, 0] for correct puck yaw observation
        self.client.resetBasePositionAndOrientation(self._model_map['puck'], puck_pos, [0, 0, 1, 0])

        for i, (model_id, joint_id, _) in enumerate(self._indexer.action_data):
            self.client.resetJointState(model_id, joint_id, self.init_state[i])

        self.has_defend = False
        self.has_hit = False
        self.has_bounce = False


    def reward(self, state, action, next_state, absorbing):
        r = 0
        puck_pos = self.get_sim_state(next_state, "puck", PyBulletObservationType.BODY_POS)[:3]
        puck_vel = self.get_sim_state(next_state, "puck", PyBulletObservationType.BODY_LIN_VEL)[:3]
        # This checks weather the puck is in our goal, heavy penalty if it is.
        # If absorbing the puck is out of bounds of the table.
        if absorbing:
            # puck position is behind table going to the negative side
            if puck_pos[0] + self.env_spec['table']['length'] / 2 < 0 and \
                    np.abs(puck_pos[1]) - self.env_spec['table']['goal'] < 0:
                r = -50
        else:
            # If the puck bounced off the head walls, there is no reward.
            if self.has_bounce:
                r = -1
            elif self.has_defend:
                # Reward if the puck slows down on the defending side
                if -0.8 < puck_pos[0] < -0.4:
                    r_y = 3 * np.exp(-3 * np.abs(puck_pos[1]))
                    r_x = np.exp(-5 * np.abs(puck_pos[0] + 0.6))
                    r_vel = 5 * np.exp(-(5 * np.linalg.norm(puck_vel)) ** 2)
                    r = r_x + r_y + r_vel + 1

                # If we did not yet hit the puck, reward is controlled by the distance between end effector and puck
                # on the x axis
            else:
                ee_pos = self.get_sim_state(next_state, "planar_robot_1/link_striker_ee",
                                            PyBulletObservationType.LINK_POS)[:2]
                ee_des = np.array([-0.6, puck_pos[1]])
                dist_ee_puck = np.abs(ee_des - ee_pos[:2])

                r_x = np.exp(-3 * dist_ee_puck[0])

                sig = 0.2
                r_y = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((dist_ee_puck[1] - 0.08) / sig, 2.) / 2)
                r = 0.3 * r_x + 0.7 * (r_y / 2)

        # penalizes the amount of torque used
        r -= self.action_penalty * np.linalg.norm(action)
        return r

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
        obs_idx = [0, 1, 7, 8, 9, 13, 14, 15, 16, 17, 18]
        obs_low = mdp_info.observation_space.low[obs_idx]
        obs_high = mdp_info.observation_space.high[obs_idx]
        obs_low[0:2] = [-1, -0.5]
        obs_high[0:2] = [1, 0.5]
        observation_space = Box(low=obs_low, high=obs_high)

        action_low = mdp_info.action_space.low[:3]
        action_high = mdp_info.action_space.high[:3]
        action_space = Box(low=action_low, high=action_high)

        return MDPInfo(observation_space, action_space, mdp_info.gamma, mdp_info.horizon)

    def is_absorbing(self, state):
        puck_pos_x = self.get_sim_state(state, "puck", PyBulletObservationType.BODY_POS)[0]
        if super().is_absorbing(state):
            return True
        return self.has_bounce

    def _simulation_post_step(self):
        if not self.has_defend:
            collision_count = len(self.client.getContactPoints(self._model_map['puck'],
                                                               self._indexer.link_map['planar_robot_1/'
                                                                                      'link_striker_ee'][0],
                                                               -1,
                                                               self._indexer.link_map['planar_robot_1/'
                                                                                      'link_striker_ee'][1]))
            if collision_count > 0:
                self.has_defend = True

        if not self.has_hit:
            collision_count = len(self.client.getContactPoints(self._model_map['puck'],
                                                               self._indexer.link_map['planar_robot_2/'
                                                                                      'link_striker_ee'][0],
                                                               -1,
                                                               self._indexer.link_map['planar_robot_2/'
                                                                                      'link_striker_ee'][1]))
            if collision_count > 0:
                self.has_hit = True

        if not self.has_bounce:
            collision_count = 0
            collision_count += len(self.client.getContactPoints(self._model_map['puck'],
                                                                self._indexer.link_map['t_up_rim_l'][0],
                                                                -1,
                                                                self._indexer.link_map['t_up_rim_l'][1]))
            collision_count += len(self.client.getContactPoints(self._model_map['puck'],
                                                                self._indexer.link_map['t_up_rim_r'][0],
                                                                -1,
                                                                self._indexer.link_map['t_up_rim_r'][1]))

            collision_count += len(self.client.getContactPoints(self._model_map['puck'],
                                                                self._indexer.link_map['t_down_rim_l'][0],
                                                                -1,
                                                                self._indexer.link_map['t_down_rim_l'][1]))
            collision_count += len(self.client.getContactPoints(self._model_map['puck'],
                                                                self._indexer.link_map['t_down_rim_r'][0],
                                                                -1,
                                                                self._indexer.link_map['t_down_rim_r'][1]))

            if collision_count > 0:
                self.has_bounce = True


if __name__ == '__main__':
    env = AirHockeyDefendHit(debug_gui=True, env_noise=False, obs_noise=False, obs_delay=False, n_intermediate_steps=4,
                       table_boundary_terminate=True, random_init=True)

    env.reset()
    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    while True:
        # action = np.random.randn(3) * 5
        action = np.array([0] * 6)
        observation, reward, done, info = env.step(action)
        gamma *= env.info.gamma
        J += gamma * reward
        R += reward
        steps += 1
        if done or steps > env.info.horizon * 5:
            print("J: ", J, " R: ", R)
            R = 0.
            J = 0.
            gamma = 1.
            steps = 0
            env.reset()
        time.sleep(1 / 60.)
