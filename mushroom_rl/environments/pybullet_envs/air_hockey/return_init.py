import time
import math
import pybullet as p
from matplotlib import image
import numpy as np
from mushroom_rl.environments.pybullet_envs.air_hockey.single import AirHockeySingle, PyBulletObservationType


class AirHockeyReturn(AirHockeySingle):
    """
    Class for the air hockey hitting task.
    The agent tries to get close to the puck if the hitting does not happen.
    And will get bonus reward if the robot scores a goal.
    """

    def __init__(self, gamma=0.99, horizon=120, env_noise=False, obs_noise=False, obs_delay=False, torque_control=True,
                 step_action_function=None, timestep=1 / 240., n_intermediate_steps=1, debug_gui=False,
                 action_penalty=1e-3):
        """
        Constructor

        Args:
            random_init(bool, False): If true, initialize the puck at random position.
            action_penalty(float, 1e-3): The penalty of the action on the reward at each time step
        """
        self.action_penalty = action_penalty
        self.ee_spawn_range = np.array([[-0.94, -0.3], [-0.47, 0.47]])

        self.puck_spawn_range = np.array([[-0.94, 0.94], [-0.47, 0.47]])

        self.desired_conf = np.array([-0.9273, 0.9273, np.pi / 2])

        self.has_hit = False

        super().__init__(gamma=gamma, horizon=horizon, timestep=timestep, n_intermediate_steps=n_intermediate_steps,
                         debug_gui=debug_gui, env_noise=env_noise, obs_noise=obs_noise, obs_delay=obs_delay,
                         torque_control=torque_control, step_action_function=step_action_function)


    def setup(self, state):
        self.puck_pos = np.random.rand(2) * (self.puck_spawn_range[:, 1] - self.puck_spawn_range[:, 0]) + self.puck_spawn_range[:, 0]

        puck_pos = np.concatenate([self.puck_pos, [-0.189]])
        self.client.resetBasePositionAndOrientation(self._model_map['puck'], puck_pos, [0, 0, 0, 1.0])

        puck_lin_vel = np.random.uniform(-1, 1, 3)
        puck_lin_vel[2] = 0.0
        puck_ang_vel = np.random.uniform(-1, 1, 3)
        puck_ang_vel[:2] = 0.0

        self.client.resetBaseVelocity(self._model_map['puck'], puck_lin_vel, puck_ang_vel)

        self.spawn_pos = np.random.rand(2) * (self.ee_spawn_range[:, 1] - self.ee_spawn_range[:, 0]) + self.ee_spawn_range[:, 0]

        robot_id, joint_id = self._indexer.link_map['planar_robot_1/link_striker_ee']
        self.init_state = self.client.calculateInverseKinematics(robot_id, joint_id, np.concatenate([self.spawn_pos, [-0.189]]))

        for i, (model_id, joint_id, _) in enumerate(self._indexer.action_data):
            self.client.resetJointState(model_id, joint_id, self.init_state[i])

        self.has_hit = False


    def reward(self, state, action, next_state, absorbing):
        robot_conf = [self.get_sim_state(next_state, 'planar_robot_1/joint_' + str(i), PyBulletObservationType.JOINT_POS)[0]
                      for i in range(1, 4)]

        r = -2
        if not self.has_hit:
            r = -np.linalg.norm(self.desired_conf - robot_conf)
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



if __name__ == '__main__':
    env = AirHockeyReturn(debug_gui=True, env_noise=False, obs_noise=False, obs_delay=False, n_intermediate_steps=4)

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
