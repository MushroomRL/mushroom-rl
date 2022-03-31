import time
import math
import pybullet as p
from matplotlib import image
import numpy as np
from mushroom_rl.environments.pybullet_envs.air_hockey.single import AirHockeySingle, PyBulletObservationType
from mushroom_rl.utils.spaces import Box
from mushroom_rl.core import MDPInfo



class AirHockeyHit(AirHockeySingle):
    """
    Class for the air hockey hitting task.
    The agent tries to get close to the puck if the hitting does not happen.
    And will get bonus reward if the robot scores a goal.
    """

    def __init__(self, gamma=0.99, horizon=120, env_noise=False, obs_noise=False, obs_delay=False, torque_control=True,
                 step_action_function=None, timestep=1 / 240., n_intermediate_steps=1, debug_gui=False,
                 random_init=False, action_penalty=1e-3, table_boundary_terminate=False, init_state="right"):
        """
        Constructor

        Args:
            random_init(bool, False): If true, initialize the puck at random position.
            action_penalty(float, 1e-3): The penalty of the action on the reward at each time step
        """
        self.hit_range = np.array([[-0.65, -0.25], [-0.4, 0.4]])
        self.goal = np.array([0.98, 0])
        self.has_hit = False
        self.has_bounce = False
        self.vel_hit_x = 0.
        self.r_hit = 0.
        self.random_init = random_init
        self.action_penalty = action_penalty
        self.init_strat = init_state
        self.vec_puck_goal = None
        self.vec_puck_side = None
        self.vec_side_goal = None
        self.side_point = None
        self.puck_pos = None
        super().__init__(gamma=gamma, horizon=horizon, timestep=timestep, n_intermediate_steps=n_intermediate_steps,
                         debug_gui=debug_gui, env_noise=env_noise, obs_noise=obs_noise, obs_delay=obs_delay,
                         torque_control=torque_control, step_action_function=step_action_function,
                         table_boundary_terminate=table_boundary_terminate)


    def setup(self, state):
        if self.random_init:
            self.puck_pos = np.random.rand(2) * (self.hit_range[:, 1] - self.hit_range[:, 0]) + self.hit_range[:, 0]
        else:
            self.puck_pos = np.mean(self.hit_range, axis=1)

        if self.init_strat == 'right':
            self.init_state = np.array([-0.9273, 0.9273, np.pi / 2])
        elif self.init_strat == 'left':
            self.init_state = -1 * np.array([-0.9273, 0.9273, np.pi / 2])
        elif self.init_strat == 'random':
            robot_id, joint_id = self._indexer.link_map['planar_robot_1/link_striker_ee']
            striker_pos_y = np.random.rand() * 0.8 - 0.4
            self.init_state = self.client.calculateInverseKinematics(robot_id, joint_id, [-0.81, striker_pos_y, -0.179])

        puck_pos = np.concatenate([self.puck_pos, [-0.189]])
        self.client.resetBasePositionAndOrientation(self._model_map['puck'], puck_pos, [0, 0, 0, 1.0])

        self.vec_puck_goal = (self.goal - self.puck_pos) / np.linalg.norm(self.goal - self.puck_pos)

        # width of table minus radius of puck
        height = 0.51 - 0.03165

        # get to point
        w = (abs(puck_pos[1]) * self.goal[0] + self.goal[1] * puck_pos[0] - height * puck_pos[0] - height *
             self.goal[0]) / (abs(puck_pos[1]) + self.goal[1] - 2 * height)

        self.side_point = np.array([w, np.copysign(height, puck_pos[1])])

        self.vec_puck_side = (self.side_point - self.puck_pos) / np.linalg.norm(self.side_point - self.puck_pos)

        self.vec_side_goal = (self.goal - self.side_point) / np.linalg.norm(self.goal - self.side_point)

        for i, (model_id, joint_id, _) in enumerate(self._indexer.action_data):
            self.client.resetJointState(model_id, joint_id, self.init_state[i])

        self.has_hit = False
        self.has_bounce = False
        self.vel_hit_x = 0.
        self.r_hit = 0.

    def reward(self, state, action, next_state, absorbing):
        r = 0
        puck_pos = self.get_sim_state(next_state, "puck", PyBulletObservationType.BODY_POS)[:2]
        # If puck is out of bounds
        if absorbing:
            # If puck is in the enemy goal
            if puck_pos[0] - self.env_spec['table']['length'] / 2 > 0 and \
                    np.abs(puck_pos[1]) - self.env_spec['table']['goal'] < 0:
                r = 200

            # If mallet hits walls, not used with safe exploration
            if self.table_boundary_terminate:
                ee_pos = self.get_sim_state(next_state, "planar_robot_1/link_striker_ee",
                                            PyBulletObservationType.LINK_POS)[:3]
                if abs(ee_pos[0]) > self.env_spec['table']['length'] / 2 or \
                        abs(ee_pos[1]) > self.env_spec['table']['width'] / 2:
                    r = -10
        else:
            if not self.has_hit:
                ee_pos = self.get_sim_state(next_state, "planar_robot_1/link_striker_ee",
                                            PyBulletObservationType.LINK_POS)[:2]
                dist_ee_puck = np.linalg.norm(puck_pos - ee_pos)

                vec_ee_puck = (puck_pos - ee_pos) / dist_ee_puck

                cos_ang_side = np.clip(self.vec_puck_side @ vec_ee_puck, 0, 1)

                # Reward if vec_ee_puck and vec_puck_goal have the same direction
                cos_ang_goal = np.clip(self.vec_puck_goal @ vec_ee_puck, 0, 1)
                cos_ang = np.max([cos_ang_goal, cos_ang_side])

                r = np.exp(-8 * (dist_ee_puck - 0.08)) * cos_ang ** 2
                # self.r_hit = r
                self.r_hit = 1
            else:
                r_hit = 0.25 + self.r_hit * min([1, (0.25 * self.vel_hit_x ** 4)])

                r_goal = 0
                if puck_pos[0] > 0.7:
                    sig = 0.1
                    r_goal = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((puck_pos[1] - 0) / sig, 2.) / 2)

                r = 2 * r_hit + 10 * r_goal

        r -= self.action_penalty * np.linalg.norm(action)
        return r

    def is_absorbing(self, state):
        if super().is_absorbing(state):
            return True
        if self.has_hit:
            puck_vel = self.get_sim_state(self._state, "puck", PyBulletObservationType.BODY_LIN_VEL)[:2]
            if np.linalg.norm(puck_vel) < 0.01:
                return True
        return self.has_bounce

    def _simulation_post_step(self):
        if not self.has_hit:
            # Kinda bad
            puck_vel = self.get_sim_state(self._state, "puck", PyBulletObservationType.BODY_LIN_VEL)[:2]
            if np.linalg.norm(puck_vel) > 0.1:
                self.has_hit = True
                self.vel_hit_x = puck_vel[0]

        if not self.has_bounce:
            # check if bounced beside the goal
            collision_count = 0
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

    def _modify_mdp_info(self, mdp_info):
        info = super(AirHockeyHit, self)._modify_mdp_info(mdp_info)
        obs_low = np.append(mdp_info.observation_space.low, [0])
        obs_high = np.append(mdp_info.observation_space.high, [1])
        observation_space = Box(low=obs_low, high=obs_high)
        return MDPInfo(observation_space, info.action_space, info.gamma, info.horizon)

    def _create_observation(self, state):
        obs = super(AirHockeyHit, self)._create_observation(state)
        return np.append(obs, [self.has_hit])


if __name__ == '__main__':
    env = AirHockeyHit(debug_gui=True, env_noise=False, obs_noise=False, obs_delay=False, n_intermediate_steps=4,
                       table_boundary_terminate=True, random_init=True, init_state="right")

    env.reset()
    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    while True:
        # action = np.random.randn(3) * 5
        action = np.array([0] * 3)
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
