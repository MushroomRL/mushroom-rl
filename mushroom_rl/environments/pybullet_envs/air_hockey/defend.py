import numpy as np

from mushroom_rl.environments.pybullet_envs.air_hockey.single import AirHockeySingleBullet, \
    PyBulletObservationType


class AirHockeyDefendBullet(AirHockeySingleBullet):
    """
    Class for the air hockey defending task.
    The agent tries to stop the puck at the line x=-0.6.
    If the puck get into the goal, it will get a punishment.
    """
    def __init__(self, gamma=0.99, horizon=500, env_noise=False, obs_noise=False, obs_delay=False, torque_control=True,
                 step_action_function=None, timestep=1 / 240., n_intermediate_steps=1, debug_gui=False,
                 random_init=False, action_penalty=1e-3, table_boundary_terminate=False, init_velocity_range=(1, 2.2)):
        """
        Constructor

        Args:
            random_init(bool, False): If true, initialize the puck at random position .
            action_penalty(float, 1e-3): The penalty of the action on the reward at each time step
            init_velocity_range((float, float), (1, 2.2)): The range in which the initial velocity is initialized
        """

        self.random_init = random_init
        self.action_penalty = action_penalty
        self.init_velocity_range = init_velocity_range

        self.start_range = np.array([[0.25, 0.65], [-0.4, 0.4]])
        self.has_hit = False
        self.has_bounce = False
        self.puck_pos = None

        super().__init__(gamma=gamma, horizon=horizon, timestep=timestep, n_intermediate_steps=n_intermediate_steps,
                         debug_gui=debug_gui, env_noise=env_noise, obs_noise=obs_noise, obs_delay=obs_delay,
                         torque_control=torque_control, step_action_function=step_action_function,
                         table_boundary_terminate=table_boundary_terminate, number_flags=2)

    def setup(self, state=None):
        # Set initial puck parameters
        if self.random_init:
            puck_pos = np.random.rand(2) * (self.start_range[:, 1] - self.start_range[:, 0]) + self.start_range[:, 0]
            puck_pos = np.concatenate([puck_pos, [-0.189]])

            lin_vel = np.random.uniform(self.init_velocity_range[0], self.init_velocity_range[1])
            angle = np.random.uniform(-0.5, 0.5)

            puck_lin_vel = np.zeros(3)
            puck_lin_vel[0] = -np.cos(angle) * lin_vel
            puck_lin_vel[1] = np.sin(angle) * lin_vel
            puck_lin_vel[2] = 0.0
            puck_ang_vel = np.random.uniform(-1, 1, 3)
            puck_ang_vel[:2] = 0.0

            # Used for data logging in eval, HAS to be puck_pos
            self.puck_pos = [puck_pos, puck_lin_vel, puck_ang_vel]
        else:
            puck_pos = np.array([self.start_range[0].mean(), 0])
            puck_pos = np.concatenate([puck_pos, [-0.189]])
            puck_lin_vel = np.array([-1., 0., 0.])
            puck_ang_vel = np.zeros(3)

        self.client.resetBasePositionAndOrientation(self._model_map['puck'], puck_pos, [0, 0, 0, 1.0])
        self.client.resetBaseVelocity(self._model_map['puck'], puck_lin_vel, puck_ang_vel)

        for i, (model_id, joint_id, _) in enumerate(self._indexer.action_data):
            self._client.resetJointState(model_id, joint_id, self.init_state[i])

        self.has_hit = False
        self.has_bounce = False

    def reward(self, state, action, next_state, absorbing):
        r = 0
        puck_pos = self.get_sim_state(next_state, "puck", PyBulletObservationType.BODY_POS)[:3]
        puck_vel = self.get_sim_state(next_state, "puck", PyBulletObservationType.BODY_LIN_VEL)[:3]

        # If absorbing the puck is out of bounds of the table.
        if absorbing:
            # large penalty if agent coincides a goal
            if puck_pos[0] + self.env_spec['table']['length'] / 2 < 0 and \
                    np.abs(puck_pos[1]) - self.env_spec['table']['goal'] < 0:
                r = -50
        else:
            # If the puck bounced off the head walls, there is no reward.
            if self.has_bounce:
                r = -1
            elif self.has_hit:
                # Reward if the puck slows down on the defending side
                if -0.8 < puck_pos[0] < -0.4:
                    r_y = 3 * np.exp(-3 * np.abs(puck_pos[1]))
                    r_x = np.exp(-5 * np.abs(puck_pos[0] + 0.6))
                    r_vel = 5 * np.exp(-(5 * np.linalg.norm(puck_vel))**2)
                    r = r_x + r_y + r_vel + 1

                # If we did not yet hit the puck, reward is controlled by the distance between end effector and puck
                # on the x axis
            else:
                ee_pos = self.get_sim_state(next_state, "planar_robot_1/link_striker_ee",
                                                PyBulletObservationType.LINK_POS)[:2]

                # Maybe change -0.6 to -0.4 so the puck is stopped a bit higher, could improve performance because
                # we don't run into the constraints at the bottom
                ee_des = np.array([-0.6, puck_pos[1]])
                dist_ee_puck = np.abs(ee_des - ee_pos[:2])

                r_x = np.exp(-3 * dist_ee_puck[0])

                sig = 0.2
                r_y = 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((dist_ee_puck[1] - 0.08)/sig, 2.)/2)
                r = 0.3 * r_x + 0.7 * (r_y/2)

        # penalizes the amount of torque used
        r -= self.action_penalty * np.linalg.norm(action)
        return r

    def is_absorbing(self, state):
        puck_pos_y = self.get_sim_state(state, "puck", PyBulletObservationType.BODY_POS)[0]
        if super().is_absorbing(state):
            return True
        if (self.has_hit or self.has_bounce) and puck_pos_y > -0.3:
            return True
        return False

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

    def _create_observation(self, state):
        obs = super(AirHockeyDefendBullet, self)._create_observation(state)
        return np.append(obs, [self.has_hit, self.has_bounce])


if __name__ == '__main__':
    import time

    env = AirHockeyDefendBullet(debug_gui=True, obs_noise=False, obs_delay=False, n_intermediate_steps=4, random_init=True)

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
