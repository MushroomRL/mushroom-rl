import time
import numpy as np
import pybullet
from mushroom_rl.environments.pybullet import PyBullet, PyBulletObservationType

from pathlib import Path

from mushroom_rl.environments.pybullet_envs import __file__ as path_robots


class HexapodBullet(PyBullet):
    def __init__(self, gamma=0.99, horizon=1000, goal=None, debug_gui=False):
        hexapod_path = Path(path_robots).absolute().parent / 'data' / 'hexapod'/ 'hexapod.urdf'
        self.robot_path = str(hexapod_path)

        action_spec = [
            ("hexapod/leg_0/joint_0", pybullet.VELOCITY_CONTROL),
            ("hexapod/leg_0/joint_1", pybullet.VELOCITY_CONTROL),
            ("hexapod/leg_0/joint_2", pybullet.VELOCITY_CONTROL),

            ("hexapod/leg_1/joint_0", pybullet.VELOCITY_CONTROL),
            ("hexapod/leg_1/joint_1", pybullet.VELOCITY_CONTROL),
            ("hexapod/leg_1/joint_2", pybullet.VELOCITY_CONTROL),

            ("hexapod/leg_2/joint_0", pybullet.VELOCITY_CONTROL),
            ("hexapod/leg_2/joint_1", pybullet.VELOCITY_CONTROL),
            ("hexapod/leg_2/joint_2", pybullet.VELOCITY_CONTROL),

            ("hexapod/leg_3/joint_0", pybullet.VELOCITY_CONTROL),
            ("hexapod/leg_3/joint_1", pybullet.VELOCITY_CONTROL),
            ("hexapod/leg_3/joint_2", pybullet.VELOCITY_CONTROL),

            ("hexapod/leg_4/joint_0", pybullet.VELOCITY_CONTROL),
            ("hexapod/leg_4/joint_1", pybullet.VELOCITY_CONTROL),
            ("hexapod/leg_4/joint_2", pybullet.VELOCITY_CONTROL),

            ("hexapod/leg_5/joint_0", pybullet.VELOCITY_CONTROL),
            ("hexapod/leg_5/joint_1", pybullet.VELOCITY_CONTROL),
            ("hexapod/leg_5/joint_2", pybullet.VELOCITY_CONTROL),
        ]

        observation_spec = [
            ("hexapod/leg_0/joint_0", PyBulletObservationType.JOINT_POS),
            ("hexapod/leg_0/joint_1", PyBulletObservationType.JOINT_POS),
            ("hexapod/leg_0/joint_2", PyBulletObservationType.JOINT_POS),

            ("hexapod/leg_1/joint_0", PyBulletObservationType.JOINT_POS),
            ("hexapod/leg_1/joint_1", PyBulletObservationType.JOINT_POS),
            ("hexapod/leg_1/joint_2", PyBulletObservationType.JOINT_POS),

            ("hexapod/leg_2/joint_0", PyBulletObservationType.JOINT_POS),
            ("hexapod/leg_2/joint_1", PyBulletObservationType.JOINT_POS),
            ("hexapod/leg_2/joint_2", PyBulletObservationType.JOINT_POS),

            ("hexapod/leg_3/joint_0", PyBulletObservationType.JOINT_POS),
            ("hexapod/leg_3/joint_1", PyBulletObservationType.JOINT_POS),
            ("hexapod/leg_3/joint_2", PyBulletObservationType.JOINT_POS),

            ("hexapod/leg_4/joint_0", PyBulletObservationType.JOINT_POS),
            ("hexapod/leg_4/joint_1", PyBulletObservationType.JOINT_POS),
            ("hexapod/leg_4/joint_2", PyBulletObservationType.JOINT_POS),

            ("hexapod/leg_5/joint_0", PyBulletObservationType.JOINT_POS),
            ("hexapod/leg_5/joint_1", PyBulletObservationType.JOINT_POS),
            ("hexapod/leg_5/joint_2", PyBulletObservationType.JOINT_POS),

            ("hexapod", PyBulletObservationType.BODY_POS),
            ("hexapod", PyBulletObservationType.BODY_LIN_VEL)
        ]

        files = {
            self.robot_path: dict(basePosition=[0.0, 0, 0.12],
                                  baseOrientation=[0, 0, 0.0, 1.0],
                                  flags=pybullet.URDF_USE_SELF_COLLISION),
            'plane.urdf': {}
        }

        super().__init__(files, action_spec, observation_spec, gamma, horizon, n_intermediate_steps=8, debug_gui=debug_gui,
                         distance=3, origin=[0., 0., 0.], angles=[0., -45., 0.])

        self._client.setGravity(0, 0, -9.81)

        self.hexapod_initial_state = np.array(
            [-0.66, 0.66, -1.45,
             0.66, -0.66, 1.45,
             0.00, 0.66, -1.45,
             0.00, -0.66, 1.45,
             0.66, 0.66, -1.45,
             -0.66, -0.66, 1.45]
        )

        self._goal = np.array([2.0, 2.0]) if goal is None else goal

    def setup(self):
        for i, (model_id, joint_id, _) in enumerate(self._action_data):
            self._client.resetJointState(model_id, joint_id, self.hexapod_initial_state[i])

        self._client.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0.0, cameraPitch=-45,
                                                cameraTargetPosition=[0., 0., 0.])

        for model_id, link_id in self._link_map.values():
            self._client.changeDynamics(model_id, link_id, lateralFriction=1.0, spinningFriction=1.0)

        for model_id in self._model_map.values():
            self._client.changeDynamics(model_id, -1, lateralFriction=1.0, spinningFriction=1.0)

        self._filter_collisions()

    def reward(self, state, action, next_state):

        pose = self.get_sim_state(next_state, "hexapod", PyBulletObservationType.BODY_POS)
        euler = pybullet.getEulerFromQuaternion(pose[3:])

        goal_distance = np.linalg.norm(pose[:2] - self._goal)
        goal_reward = np.exp(-goal_distance)

        attitude_distance = np.linalg.norm(euler[:2])
        attitude_reward = np.exp(-attitude_distance)

        action_penalty = np.linalg.norm(action)

        self_collisions_penalty = 1.0*self._count_self_collisions()

        return 1 + goal_reward + 1e-1*attitude_reward - 1e-3*action_penalty - self_collisions_penalty

    def is_absorbing(self, state):
        pose = self.get_sim_state(state, "hexapod", PyBulletObservationType.BODY_POS)

        euler = pybullet.getEulerFromQuaternion(pose[3:])

        return pose[2] > 0.2 or abs(euler[0]) > np.pi/4 or abs(euler[1]) > np.pi/4 or self._count_self_collisions() >= 2

    def _count_self_collisions(self):
        hexapod_id = self._model_map['hexapod']

        collision_count = 0
        collisions = self._client.getContactPoints(hexapod_id)

        for collision in collisions:
            body_2 = collision[2]
            if body_2 == hexapod_id:
                collision_count += 1

        return collision_count

    def _filter_collisions(self):
        # Disable fixed links collisions
        for leg_n in range(6):
            for link_n in range(3):
                motor_name = f'hexapod/leg_{leg_n}/motor_{link_n}'
                link_name = f'hexapod/leg_{leg_n}/link_{link_n}'
                self._client.setCollisionFilterPair(self._link_map[motor_name][0], self._link_map[link_name][0],
                                                    self._link_map[motor_name][1], self._link_map[link_name][1], 0)


if __name__ == '__main__':
    from mushroom_rl.core import Core
    from mushroom_rl.core import Agent
    from mushroom_rl.utils.dataset import compute_J


    class DummyAgent(Agent):
        def __init__(self, n_actions):
            self._n_actions = n_actions

        def draw_action(self, state):
            time.sleep(0.01)

            return np.random.randn(self._n_actions)

        def episode_start(self):
            pass

        def fit(self, dataset):
            pass


    mdp = HexapodBullet(debug_gui=True)
    agent = DummyAgent(mdp.info.action_space.shape[0])

    core = Core(agent, mdp)
    dataset = core.evaluate(n_episodes=10, render=False)
    print('reward: ', compute_J(dataset, mdp.info.gamma))
    print("mdp_info state shape", mdp.info.observation_space.shape)
    print("actual state shape", dataset[0][0].shape)
    print("mdp_info action shape", mdp.info.action_space.shape)
    print("actual action shape", dataset[0][1].shape)

    print("action low", mdp.info.action_space.low)
    print("action high", mdp.info.action_space.high)
