import numpy as np

from mushroom_rl.environments.mujoco_envs.air_hockey.single import AirHockeySingle


class AirHockeyRepel(AirHockeySingle):
    """
    Class for the air hockey repel task.
    The agent tries repel the puck to the opponent.
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
        self.goal = np.array([0.98, 0])

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

        super(AirHockeyRepel, self).setup(obs)

    # Very flawed needs a lot of tuning
    def reward(self, state, action, next_state, absorbing):
        r = 0
        puck_pos, puck_vel, _ = self.get_puck(next_state)

        # If absorbing the puck is out of bounds of the table.
        if absorbing:
            # big penalty if we coincide a goal
            if puck_pos[0] + self.env_spec['table']['length'] / 2 < 0 and \
                    np.abs(puck_pos[1]) - self.env_spec['table']['goal'] < 0:
                r = -50
        else:
            if self.has_hit:

                r_x = puck_pos[0] + 0.98
                r_vel = min([puck_vel[0] ** 3, 5])

                r = r_x + r_vel + 1

                if puck_pos[0] > 0.9:
                    r += 100 * np.exp(-3 * abs(puck_pos[1]))
                # If we did not yet hit the puck, reward is controlled by the distance between end effector and puck
                # on the x axis
            else:
                ee_pos = self.get_ee()[0][:2]

                ee_des = np.array([-0.6, puck_pos[1]])
                dist_ee_puck = np.abs(ee_des - ee_pos)

                r_x = np.exp(-3 * dist_ee_puck[0])

                sig = 0.2
                r_y = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((dist_ee_puck[1] - 0.08) / sig, 2.) / 2)
                r = 0.3 * r_x + 0.7 * (r_y / 2)

        # penalizes the amount of torque used
        r -= self.action_penalty * np.linalg.norm(action)
        return r

    # If the Puck is out of Bounds of the table this returns True
    def is_absorbing(self, state):
        if super().is_absorbing(state):
            return True
        return self.has_bounce
