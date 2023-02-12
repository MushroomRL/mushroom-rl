import numpy as np
from mushroom_rl.environments.mujoco_envs.air_hockey.single import AirHockeySingle


class AirHockeyHit(AirHockeySingle):
    """
    Class for the air hockey hitting task.
    The agent tries to get close to the puck if the hitting does not happen.
    And will get bonus reward if the robot scores a goal.
    """

    def __init__(self, random_init=False, action_penalty=1e-3, init_robot_state="right", gamma=0.99, horizon=120,
                 env_noise=False, obs_noise=False, timestep=1 / 240., n_intermediate_steps=1, **viewer_params):
        """
        Constructor

        Args:
            random_init(bool, False): If true, initialize the puck at random position.
            action_penalty(float, 1e-3): The penalty of the action on the reward at each time step
            init_robot_state(string, "right"): The configuration in which the robot is initialized. "right", "left",
                "random" available.

        """

        self.random_init = random_init
        self.action_penalty = action_penalty
        self.init_robot_state = init_robot_state

        self.hit_range = np.array([[-0.65, -0.25], [-0.4, 0.4]])
        self.goal = np.array([0.98, 0])
        self.vec_puck_goal = None
        self.vec_puck_side = None
        super().__init__(gamma=gamma, horizon=horizon, timestep=timestep, n_intermediate_steps=n_intermediate_steps,
                         env_noise=env_noise, obs_noise=obs_noise, **viewer_params)

    def setup(self, obs):
        # Initial position of the puck
        if self.random_init:
            puck_pos = np.random.rand(2) * (self.hit_range[:, 1] - self.hit_range[:, 0]) + self.hit_range[:, 0]
        else:
            puck_pos = np.mean(self.hit_range, axis=1)

        # Initial configuration of the robot arm

        if self.init_robot_state == 'right':
            self.init_state = np.array([-0.9273, 0.9273, np.pi / 2])
        elif self.init_robot_state == 'left':
            self.init_state = -1 * np.array([-0.9273, 0.9273, np.pi / 2])

        self._write_data("puck_pos", np.concatenate([puck_pos, [0, 0, 0, 0, 1]]))

        self.vec_puck_goal = (self.goal - puck_pos) / np.linalg.norm(self.goal - puck_pos)

        # width of table minus radius of puck
        effective_width = 0.51 - 0.03165

        # Calculate bounce point by assuming incoming angle = outgoing angle
        w = (abs(puck_pos[1]) * self.goal[0] + self.goal[1] * puck_pos[0] - effective_width * puck_pos[
            0] - effective_width *
             self.goal[0]) / (abs(puck_pos[1]) + self.goal[1] - 2 * effective_width)

        side_point = np.array([w, np.copysign(effective_width, puck_pos[1])])

        self.vec_puck_side = (side_point - puck_pos) / np.linalg.norm(side_point - puck_pos)

        super(AirHockeyHit, self).setup(obs)

    def reward(self, state, action, next_state, absorbing):
        r = 0
        puck_pos, puck_vel, _ = self.get_puck(next_state)

        # If puck is out of bounds
        if absorbing:
            # If puck is in the opponent goal
            if (puck_pos[0] - self.env_spec['table']['length'] / 2) > 0 and \
                    (np.abs(puck_pos[1]) - self.env_spec['table']['goal']) < 0:
                r = 200

        else:
            if not self.has_hit:
                ee_pos = self.get_ee()[0][:2]

                dist_ee_puck = np.linalg.norm(puck_pos[:2] - ee_pos)

                vec_ee_puck = (puck_pos[:2] - ee_pos) / dist_ee_puck

                cos_ang_side = np.clip(self.vec_puck_side @ vec_ee_puck, 0, 1)

                # Reward if vec_ee_puck and vec_puck_goal have the same direction
                cos_ang_goal = np.clip(self.vec_puck_goal @ vec_ee_puck, 0, 1)
                cos_ang = np.max([cos_ang_goal, cos_ang_side])

                r = np.exp(-8 * (dist_ee_puck - 0.08)) * cos_ang ** 2
            else:
                r_hit = 0.25 + min([1, (0.25 * puck_vel[0] ** 4)])

                r_goal = 0
                if puck_pos[0] > 0.7:
                    sig = 0.1
                    r_goal = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((puck_pos[1] - 0) / sig, 2.) / 2)

                r = 2 * r_hit + 10 * r_goal

        r -= self.action_penalty * np.linalg.norm(action)
        return r


if __name__ == '__main__':
    env = AirHockeyHit(env_noise=False, obs_noise=False, n_intermediate_steps=4, random_init=True,
                       init_robot_state="right")

    env.reset()
    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    while True:
        action = np.random.randn(3) * 5
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
