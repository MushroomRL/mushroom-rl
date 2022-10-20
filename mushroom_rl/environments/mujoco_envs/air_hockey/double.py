import numpy as np

from mushroom_rl.utils.spaces import Box
from mushroom_rl.environments.mujoco_envs.air_hockey.base import AirHockeyBase


class AirHockeyDouble(AirHockeyBase):
    """
    Base class for two agents air hockey tasks.

    """
    def __init__(self, gamma=0.99, horizon=120, env_noise=False, obs_noise=False, timestep=1 / 240.,
                 n_intermediate_steps=1):

        """
        Constructor.

        """
        self.init_state = np.array([-0.9273, 0.9273, np.pi / 2])
        super().__init__(gamma=gamma, horizon=horizon, env_noise=env_noise, n_agents=2, obs_noise=obs_noise,
                         timestep=timestep, n_intermediate_steps=n_intermediate_steps)

        self.obs_helper.remove_obs("puck_pos", 2)
        self.obs_helper.remove_obs("puck_vel", 0)
        self.obs_helper.remove_obs("puck_vel", 1)
        self.obs_helper.remove_obs("puck_vel", 5)

        self.obs_helper.remove_obs("robot_2/puck_pos", 2)
        self.obs_helper.remove_obs("robot_2/puck_vel", 0)
        self.obs_helper.remove_obs("robot_2/puck_vel", 1)
        self.obs_helper.remove_obs("robot_2/puck_vel", 5)

        self.obs_helper.add_obs("collision_robot_1_puck", 1, 0, 1)
        self.obs_helper.add_obs("collision_robot_2_puck", 1, 0, 1)
        self.obs_helper.add_obs("collision_short_sides_rim_puck", 1, 0, 1)

        self._mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())

        self.robot_1_hit = False
        self.robot_2_hit = False
        self.has_bounce = False

    def get_puck(self, obs):
        puck_pos = self.obs_helper.get_from_obs(obs, "puck_pos")
        puck_lin_vel = self.obs_helper.get_from_obs(obs, "puck_vel")[1:]
        puck_ang_vel = self.obs_helper.get_from_obs(obs, "puck_vel")[0]
        return puck_pos, puck_lin_vel, puck_ang_vel

    def get_ee(self, robot=1):
        ee_pos = self._read_data("planar_robot_" + str(robot) + "/ee_pos")

        ee_vel = self._read_data("planar_robot_" + str(robot) + "/ee_vel")

        return ee_pos, ee_vel

    def _modify_observation(self, obs):
        self._puck_2d_in_robot_frame(self.obs_helper.get_from_obs(obs, "puck_pos"), self.agents[0]["frame"])

        self._puck_2d_in_robot_frame(self.obs_helper.get_from_obs(obs, "puck_vel"), self.agents[0]["frame"], type='vel')

        self._puck_2d_in_robot_frame(self.obs_helper.get_from_obs(obs, "robot_2/puck_pos"), self.agents[1]["frame"])

        self._puck_2d_in_robot_frame(self.obs_helper.get_from_obs(obs, "robot_2/puck_vel"), self.agents[1]["frame"],
                                     type='vel')

        if self.obs_noise:
            noise = np.random.randn(2) * 0.001
            self.obs_helper.get_from_obs(obs, "puck_pos")[:] += noise
            self.obs_helper.get_from_obs(obs, "robot_2/puck_pos")[:] += noise

        return obs

    def reward(self, state, action, next_state, absorbing):
        return 0

    def setup(self):
        self.robot_1_hit = False
        self.robot_2_hit = False
        self.has_bounce = False

        for i in range(3):
            self._data.joint("planar_robot_1/joint_" + str(i+1)).qpos = self.init_state[i]

        for i in range(3):
            self._data.joint("planar_robot_2/joint_" + str(i+1)).qpos = self.init_state[i]

    def _simulation_post_step(self):
        if not self.robot_1_hit:
            self.robot_1_hit = self._check_collision("puck", "robot_1/ee")

        if not self.robot_2_hit:
            self.robot_2_hit = self._check_collision("puck", "robot_1/ee")

        if not self.has_bounce:
            self.has_bounce = self._check_collision("puck", "rim_short_sides")

    def _create_observation(self, state):
        obs = super(AirHockeyDouble, self)._create_observation(state)
        return np.append(obs, [self.robot_1_hit, self.robot_2_hit, self.has_bounce])
