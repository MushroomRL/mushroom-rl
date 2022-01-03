import numpy as np
import time

from mushroom_rl.environments.pybullet_envs.air_hockey.single import AirHockeySingle, \
    PyBulletObservationType



class AirHockeyRepelle(AirHockeySingle):
    def __init__(self, gamma=0.99, horizon=500, env_noise=False, obs_noise=False, obs_delay=False, torque_control=True,
                 step_action_function=None, timestep=1 / 240., n_intermediate_steps=1, debug_gui=False,
                 random_init=False, action_penalty=1e-3):
        self.random_init = random_init
        self.action_penalty = action_penalty

        self.start_range = np.array([[0.2, 0.78], [-0.4, 0.4]])
        self.goal = np.array([0.98, 0])

        self.has_hit = False
        self.has_bounce = False
        self.vel_hit_x = 0.
        self.r_hit = 0.

        self.puck_pos = None
        super().__init__(gamma=gamma, horizon=horizon, timestep=timestep, n_intermediate_steps=n_intermediate_steps,
                         debug_gui=debug_gui, env_noise=env_noise, obs_noise=obs_noise, obs_delay=obs_delay,
                         torque_control=torque_control, step_action_function=step_action_function)

    def setup(self, state=None):
        if self.random_init:
            puck_pos = np.random.rand(2) * (self.start_range[:, 1] - self.start_range[:, 0]) + self.start_range[:, 0]
            puck_pos = np.concatenate([puck_pos, [-0.189]])

            lin_vel = 1
            angle = np.random.uniform(-0.5, 0.5)

            puck_lin_vel = np.zeros(3)
            puck_lin_vel[0] = -np.cos(angle) * lin_vel
            puck_lin_vel[1] = np.sin(angle) * lin_vel
            puck_lin_vel[2] = 0.0
            puck_ang_vel = np.random.uniform(-1, 1, 3)
            puck_ang_vel[:2] = 0.0

            self.puck_pos = [puck_pos, puck_lin_vel, puck_ang_vel]
        else:
            puck_pos = np.array([self.start_range[0].mean(), 0.0])
            puck_pos = np.concatenate([puck_pos, [-0.189]])
            puck_lin_vel = np.array([-1., 0., 0.])
            puck_ang_vel = np.zeros(3)

        self.client.resetBasePositionAndOrientation(self._model_map['puck'], puck_pos, [0, 0, 0, 1.0])
        self.client.resetBaseVelocity(self._model_map['puck'], puck_lin_vel, puck_ang_vel)

        for i, (model_id, joint_id, _) in enumerate(self._indexer.action_data):
            self._client.resetJointState(model_id, joint_id, self.init_state[i])

        self.has_hit = False
        self.has_bounce = False
        self.vel_hit_x = 0.
        self.r_hit = 0.

    # Very flawed needs a lot of tuning
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
            if self.has_hit:
                # Reward if the puck slows down on the defending side

                r_x = puck_pos[0] + 0.98
                r_vel = min([np.linalg.norm(puck_vel) ** 3, 5])

                r = r_x + r_vel + 1

                if puck_pos[0] > 0.9:
                    r += 100 * np.exp(-1 * abs(puck_pos[1]))
                # If we did not yet hit the puck, reward is controlled by the distance between end effector and puck
                # on the x axis
            else:
                ee_pos = self.get_sim_state(next_state, "planar_robot_1/link_striker_ee",
                                                PyBulletObservationType.LINK_POS)[:2]
                ee_des = np.array([-0.6, puck_pos[1]])
                dist_ee_puck = np.abs(ee_des - ee_pos[:2])

                r_x = np.exp(-3 * dist_ee_puck[0])

                sig = 0.2
                r_y = 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((dist_ee_puck[1] - 0.08)/sig, 2.)/2)
                r = 0.3 * r_x + 0.7 * (r_y/2)

        # penalizes the amount of torque used
        r -= self.action_penalty * np.linalg.norm(action)
        return r

    # If the Puck is out of Bounds of the table this returns True
    # This function is not needed???
    def is_absorbing(self, state):
        puck_pos_y = self.get_sim_state(state, "puck", PyBulletObservationType.BODY_POS)[0]
        if super().is_absorbing(state):
            return True
        return self.has_bounce

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


if __name__ == "__main__":
    env = AirHockeyRepelle(debug_gui=True, env_noise=False, obs_noise=False, obs_delay=False, n_intermediate_steps=4,
                           random_init=True)

    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    env.reset()
    while True:
        action = np.zeros(3)
        observation, reward, done, info = env.step(action)
        gamma *= env.info.gamma
        J += gamma * reward
        R += reward
        steps += 1
        if done or steps > env.info.horizon:
            print("J: ", J, " R: ", R)
            R = 0.
            J = 0.
            gamma = 1.
            steps = 0
            env.reset()
        time.sleep(1 / 60.)