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

        observation_spec = [("ball", PyBulletObservationType.BODY_POS),
                            ("ball", PyBulletObservationType.BODY_LIN_VEL),
                            ("R_palm", PyBulletObservationType.LINK_POS),
                            ("R_SFE", PyBulletObservationType.JOINT_POS),
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

        self._touched = False

        pybullet.changeDynamics(self._model_map['plane'], -1, restitution=1)
        pybullet.changeDynamics(self._model_map['table'], -1, restitution=1)
        pybullet.changeDynamics(self._model_map['ball'], -1, restitution=0.8, rollingFriction=0.01)
        pybullet.changeDynamics(self._model_map['darias'], -1, restitution=1)
        for i in range(pybullet.getNumJoints(self._model_map['darias'])):
            pybullet.changeDynamics(self._model_map['darias'], i, restitution=1)

    def setup(self):
        x_start = np.random.rand() * 0.2 + 0.4
        speed = np.random.rand() * 2 + 2
        theta = (np.random.rand() * 20 - 10) / 180 * np.pi

        # self.darias.resetJoints(self.initial_joints)
        pybullet.resetBasePositionAndOrientation(self._model_map['table'], [0.5, 0, 0.3], [0, 0, 0, 1])
        pybullet.resetBasePositionAndOrientation(self._model_map['ball'], [x_start, 1, 0.8], [0, 0, 0, 1])
        pybullet.resetBaseVelocity(self._model_map['ball'], [speed * np.sin(theta), -speed * np.cos(theta), 0],
                                   [0, 0, 0])

        self._touched = False

    def reward(self, state, action, next_state):
        if self._touched:
            R = [-1, -1, -1]
            ball_speed = state[7:10]
            reward_ball = np.dot(ball_speed**2, R)
            reward = 5e-3 * reward_ball
        else:
            ball_pos = state[:3]
            hand_pos = state[10:13]
            hand_ball_distance = np.linalg.norm(ball_pos - hand_pos)
            reward_position = -hand_ball_distance**2
            reward = 0.1 * reward_position

        if np.linalg.norm(state[7:10]) < 0.0001:
            reward = 20
        return reward

    def is_absorbing(self, state):
        ball_speed = state[7:9]
        ball_height = state[2]
        return np.linalg.norm(ball_speed) < 0.0001 or ball_height < 0.7

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

        return dict(table=table, ball=ball)

    def _step_finalize(self):

        if not self._touched:
            robot = self._model_map['darias']
            ball = self._model_map['ball']
            contact_points = pybullet.getContactPoints(bodyA=robot,bodyB=ball)

            self._touched = len(contact_points) > 0


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
    dataset = core.evaluate(n_episodes=2, render=True)
    print("mdp_info state shape", mdp.info.observation_space.shape)
    print("actual state shape", dataset[0][0].shape)
    print("mdp_info action shape", mdp.info.action_space.shape)
    print("actual action shape", dataset[0][1].shape)