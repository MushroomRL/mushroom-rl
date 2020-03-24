import os
import pybullet
import numpy as np

from mushroom_rl.environments.pybullet import PyBullet, PyBulletObservationType


class BallRolling(PyBullet):
    def __init__(self):
        robot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ball_rolling", "darias.urdf")
        action_spec = [("R_SFE", pybullet.POSITION_CONTROL),
                       ("R_SAA", pybullet.POSITION_CONTROL),
                       ("R_HR", pybullet.POSITION_CONTROL),
                       ("R_EB", pybullet.POSITION_CONTROL),
                       ("R_WR", pybullet.POSITION_CONTROL),
                       ("R_WFE", pybullet.POSITION_CONTROL),
                       ("R_WAA", pybullet.POSITION_CONTROL)
                       ]

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

        files = [
            'plane.urdf',
            robot_path
        ]

        super().__init__(files, action_spec, pybullet.POSITION_CONTROL, observation_spec, 0.99, 1000)

        plane, robot, table, ball = self._model_ids
        pybullet.changeDynamics(plane, -1, restitution=1)
        pybullet.changeDynamics(table, -1, restitution=1)
        pybullet.changeDynamics(ball, -1, restitution=0.8, rollingFriction=0.01)
        pybullet.changeDynamics(robot, -1, restitution=1)
        for i in range(pybullet.getNumJoints(robot)):
            pybullet.changeDynamics(robot, i, restitution=1)

    def setup(self):
        plane, robot, table, ball = self._model_ids

        x_start = np.random.rand() * 0.2 + 0.4
        speed = np.random.rand() * 2 + 2
        theta = (np.random.rand() * 20 - 10) / 180 * np.pi

        # self.darias.resetJoints(self.initial_joints)
        pybullet.resetBasePositionAndOrientation(table, [0.5, 0, 0.3], [0, 0, 0, 1])
        pybullet.resetBasePositionAndOrientation(ball, [x_start, 1, 0.8], [0, 0, 0, 1])
        pybullet.resetBaseVelocity(ball, [speed * np.sin(theta), -speed * np.cos(theta), 0], [0, 0, 0])

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

    def _custom_load_models(self):

        table_collision = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.5, 1.5, 0.3])
        table_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.5, 1.5, 0.3],
                                                  rgbaColor=[0.2, 1, 0.5, 1])
        table = pybullet.createMultiBody(baseMass=10.0,
                                         baseCollisionShapeIndex=table_collision,
                                         baseVisualShapeIndex=table_visual,
                                         basePosition=[0.5, 0, 0.3])

        ball_collision = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=0.2)
        ball_visual = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=0.2, rgbaColor=[0.2, 0.5, 1, 1])
        ball = pybullet.createMultiBody(baseMass=0.2,
                                        baseCollisionShapeIndex=ball_collision,
                                        baseVisualShapeIndex=ball_visual,
                                        basePosition=[0.5, 1, 0.8])

        return [table, ball]



if __name__ == '__main__':
    from mushroom_rl.core import Core
    from mushroom_rl.algorithms import Agent


    class DummyAgent(Agent):
        def __init__(self, n_actions):
            self._n_actions = n_actions

        def draw_action(self, state):
            return np.random.randn(self._n_actions)

        def episode_start(self):
            pass

        def fit(self, dataset):
            pass


    mdp = BallRolling()
    agent = DummyAgent(mdp.info.action_space.shape[0])

    core = Core(agent, mdp)
    mdp.reset()

    core.evaluate(n_episodes=3, render=True)