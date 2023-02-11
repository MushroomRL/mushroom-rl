import numpy as np

from mushroom_rl.environments.mujoco_envs.air_hockey.single import AirHockeySingle


class AirHockeyDefend(AirHockeySingle):
    """
    Class for the air hockey defending task.
    The agent tries to stop the puck at the line x=-0.6.
    If the puck get into the goal, it will get a punishment.
    """
    def __init__(self, random_init=False, action_penalty=1e-3, init_velocity_range=(1, 2.2), gamma=0.99, horizon=500,
                 env_noise=False, obs_noise=False, timestep=1 / 240., n_intermediate_steps=1, **viewer_params):
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

        super().__init__(gamma=gamma, horizon=horizon, timestep=timestep, n_intermediate_steps=n_intermediate_steps,
                         env_noise=env_noise, obs_noise=obs_noise, **viewer_params)

    def setup(self, obs):
        # Set initial puck parameters
        if self.random_init:
            puck_pos = np.random.rand(2) * (self.start_range[:, 1] - self.start_range[:, 0]) + self.start_range[:, 0]

            lin_vel = np.random.uniform(self.init_velocity_range[0], self.init_velocity_range[1])
            angle = np.random.uniform(-0.5, 0.5)

            puck_lin_vel = np.zeros(3)
            puck_lin_vel[0] = -np.cos(angle) * lin_vel
            puck_lin_vel[1] = np.sin(angle) * lin_vel

            puck_ang_vel = np.random.uniform(-1, 1, 3)
            puck_ang_vel[:2] = 0.0

        else:
            puck_pos = np.array([self.start_range[0].mean(), 0])
            puck_lin_vel = np.array([-1., 0., 0.])
            puck_ang_vel = np.zeros(3)

        self._write_data("puck_pos", np.concatenate([puck_pos, [0, 0, 0, 0, 1]]))
        self._write_data("puck_vel", np.concatenate([puck_lin_vel, puck_ang_vel]))

        super(AirHockeyDefend, self).setup(obs)

    def reward(self, state, action, next_state, absorbing):
        r = 0
        puck_pos, puck_vel, _ = self.get_puck(next_state)

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
                ee_pos = self.get_ee()[0][:2]

                # Maybe change -0.6 to -0.4 so the puck is stopped a bit higher, could improve performance because
                # we don't run into the constraints at the bottom
                ee_des = np.array([-0.6, puck_pos[1]])
                dist_ee_puck = np.abs(ee_des - ee_pos)

                r_x = np.exp(-3 * dist_ee_puck[0])

                sig = 0.2
                r_y = 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((dist_ee_puck[1] - 0.08)/sig, 2.)/2)
                r = 0.3 * r_x + 0.7 * (r_y/2)

        # penalizes the amount of torque used
        r -= self.action_penalty * np.linalg.norm(action)
        return r

    def is_absorbing(self, state):
        puck_pos_x = self.get_puck(state)[0][0]
        if (self.has_hit or self.has_bounce) and puck_pos_x > 0:
            return True
        return super().is_absorbing(state)


if __name__ == '__main__':
    env = AirHockeyDefend(obs_noise=False, n_intermediate_steps=4, random_init=True)

    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    env.reset()
    while True:
        action = np.zeros(3)
        observation, reward, done, info = env.step(action)
        env.render()
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
