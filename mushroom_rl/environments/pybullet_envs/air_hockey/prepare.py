import time

import numpy as np
from mushroom_rl.environments.pybullet_envs.air_hockey.single import AirHockeySingle, PyBulletObservationType


class AirHockeyPrepare(AirHockeySingle):
    def __init__(self, gamma=0.99, horizon=500, env_noise=False, obs_noise=False, obs_delay=False, torque_control=True,
                 step_action_function=None, timestep=1 / 240., n_intermediate_steps=1, debug_gui=False,
                 random_init=False, action_penalty=1e-3):

        self.random_init = random_init
        self.action_penalty = action_penalty

        self.desired_point = np.array([-0.6, 0])

        self.start_range = np.array([[-0.9, -0.4], [0.25, 0.48]])

        self.r_hit = None

        self.got_reward = None
        self.has_hit = False
        self.has_bounce = False
        super().__init__(gamma=gamma, horizon=horizon, timestep=timestep, n_intermediate_steps=n_intermediate_steps,
                         debug_gui=debug_gui, env_noise=env_noise, obs_noise=obs_noise, obs_delay=obs_delay,
                         torque_control=torque_control, step_action_function=step_action_function)

    def setup(self, state):
        if self.random_init:
            puck_pos = np.random.rand(2) * (self.start_range[:, 1] - self.start_range[:, 0]) + self.start_range[:, 0]
            puck_pos *= [1, [1, -1][np.random.randint(2)]]
        else:
            puck_pos = np.mean(self.start_range, axis=1)

        puck_pos = np.concatenate([puck_pos, [-0.189]])
        self.client.resetBasePositionAndOrientation(
            self._model_map['puck'], puck_pos, [0, 0, 0, 1.0])

        for i, (model_id, joint_id, _) in enumerate(self._indexer.action_data):
            self.client.resetJointState(model_id, joint_id, self.init_state[i])

        self.got_reward = None
        self.has_hit = False
        self.has_bounce = False


    def reward(self, state, action, next_state, absorbing):
        puck_pos = self.get_sim_state(next_state, "puck", PyBulletObservationType.BODY_POS)[:2]
        puck_vel = self.get_sim_state(next_state, "puck", PyBulletObservationType.BODY_LIN_VEL)[:2]

        r_vel = np.exp(-max(np.linalg.norm(puck_vel)**2 - 0.1, 0))

        if self.has_hit and not self.has_bounce:
            r = self.r_hit + r_vel

        elif self.has_bounce:
            dist_puck_des = np.linalg.norm(puck_pos - self.desired_point)
            r = 0
            if dist_puck_des < 0.2:
                if self.got_reward is None:
                    self.got_reward = True

                if self.got_reward:
                    r = 2
            else:
                if self.got_reward:
                    self.got_reward = False
        else:
            ee_pos = self.get_sim_state(next_state, "planar_robot_1/link_striker_ee",
                                        PyBulletObservationType.LINK_POS)[:2]

            dist_ee_puck = np.linalg.norm(puck_pos - ee_pos[:2])
            vec_ee_puck = (puck_pos - ee_pos) / dist_ee_puck

            height = 0.51 - 0.03165

            # get to point
            w = (abs(puck_pos[1]) * self.desired_point[0] + self.desired_point[1] * puck_pos[0] - height * puck_pos[0] -
                 height * self.desired_point[0]) / (abs(puck_pos[1]) + self.desired_point[1] - 2 * height)

            side_point = np.array([w, np.copysign(height, puck_pos[1])])

            vec_puck_side = (side_point - puck_pos) / np.linalg.norm(side_point - puck_pos)

            cos_ang = np.clip(vec_ee_puck @ vec_puck_side, 0, 1)

            r = np.exp(-3 * (dist_ee_puck - 0.08)) * cos_ang
            self.r_hit = r

        r -= self.action_penalty * np.linalg.norm(action)
        return r

    def _simulation_post_step(self):
        if not self.has_hit:
            collision_count = len(self.client.getContactPoints(self._model_map['puck'],
                                                               self._indexer.link_map['planar_robot_1/'
                                                                                      'link_striker_ee'][0],
                                                               -1,
                                                               self._indexer.link_map['planar_robot_1/'
                                                                                      'link_striker_ee'][1]))
            if collision_count > 0:
                self.has_hit = True

        if not self.has_bounce:
            collision_count = 0
            collision_count += len(self.client.getContactPoints(self._model_map['puck'],
                                                                self._indexer.link_map['t_left_rim'][0],
                                                                -1,
                                                                self._indexer.link_map['t_left_rim'][1]))
            collision_count += len(self.client.getContactPoints(self._model_map['puck'],
                                                                self._indexer.link_map['t_right_rim'][0],
                                                                -1,
                                                                self._indexer.link_map['t_right_rim'][1]))

            if collision_count > 0:
                self.has_bounce = True


if __name__ == '__main__':
    env = AirHockeyPrepare(debug_gui=True, obs_noise=False, obs_delay=False, n_intermediate_steps=4, random_init=True)

    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    env.reset()
    while True:
        # action = np.random.randn(3) * 5
        action = np.array([0, 0, 0])
        observation, reward, done, info = env.step(action)
        gamma *= env.info.gamma
        J += gamma * reward
        R += reward
        steps += 4
        if done or steps > env.info.horizon * 2:
            print("J: ", J, " R: ", R)
            R = 0.
            J = 0.
            gamma = 1.
            steps = 0
            env.reset()
        time.sleep(1 / 60.)