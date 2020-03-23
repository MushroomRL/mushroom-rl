import os
import pybullet
import numpy as np

from mushroom_rl.environments.pybullet import PyBullet, PyBulletObservationType


class BallRolling(PyBullet):
    def __init__(self):
        urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ball_rolling", "darias.urdf")
        action_spec = ["R_SFE", "R_SAA", "R_HR",  "R_EB",  "R_WR",  "R_WFE", "R_WAA"]

        observation_spec = [("R_SFE", PyBulletObservationType.JOINT_POS),
                            ("R_SFE", PyBulletObservationType.JOINT_VEL),
                            ("R_SAA", PyBulletObservationType.JOINT_POS),
                            ("R_SAA", PyBulletObservationType.JOINT_VEL),
                            ("R_HR", PyBulletObservationType.JOINT_POS),
                            ("R_HR", PyBulletObservationType.JOINT_VEL),
                            ("R_EB", PyBulletObservationType.JOINT_POS),
                            ("R_EB", PyBulletObservationType.JOINT_VEL),
                            ("R_WR", PyBulletObservationType.JOINT_POS),
                            ("R_WR", PyBulletObservationType.JOINT_VEL),
                            ("R_WFE", PyBulletObservationType.JOINT_POS),
                            ("R_WFE", PyBulletObservationType.JOINT_VEL),
                            ("R_WAA", PyBulletObservationType.JOINT_POS),
                            ("R_WAA", PyBulletObservationType.JOINT_VEL)
                            ]
        super().__init__(urdf_path, action_spec, pybullet.POSITION_CONTROL, observation_spec, 0.99, 1000)

    def reward(self, state, action, next_state):
        # R = [-1, -1, -1]
        # reward_ball = np.dot((np.array(state[4]) / self.context[1]) ** 2, R)
        # Q = [-2, -1.8, -1.5, -1.3, -1, -1, -1]
        # reward_darias = np.dot(np.array(state[1]) ** 2, Q)
        #
        # hand_ball_distance = self.getMagnitude(np.array(state[2]) - np.array(state[6]))
        # reward_position = -1 * (hand_ball_distance) ** 2
        #
        # # print(reward_ball, reward_darias, reward_position)
        #
        # # reward = 10*reward_ball+reward_darias
        # # reward = 10 * reward_ball+ 50*reward_position
        # if self.touched:
        #     reward = 5 * reward_ball
        # else:
        #     reward = 100 * reward_position
        #
        # if self.getMagnitude(state[4]) < 0.0001: #FIXME
        #     reward = 20000
        # return reward / self._maxSteps

        return 0

    def is_absorbing(self, state):
        return False
        # if (self.getMagnitude(self._observation[4]) < 0.0001):
        #     return True
        # if self._observation[2][2] < 0.7:
        #     return True
        # return False

    # def getMagnitude(self, vec):
    #     return np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)



if __name__ == '__main__':
    mdp = BallRolling()

    mdp.reset()
    absorbing = False
    while not absorbing:
        action = np.random.randn(mdp.info.action_space.shape[0])
        state, reward, absorbing, _ = mdp.step(action)
        mdp.render()