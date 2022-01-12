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


    def reward(self, state, action, next_state, absorbing):
        r = 0
        return r


    def is_absorbing(self, state):
        if super().is_absorbing(state):
            return True
        return False


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
