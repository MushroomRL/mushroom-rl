import time
import numpy as np
import pybullet
from mushroom_rl.environments.pybullet import PyBullet, PyBulletObservationType

from pathlib import Path

from mushroom_rl.environments.pybullet_envs import __file__ as path_robots


class ManipulatorBullet(PyBullet):
    def __init__(self, gamma=0.99, horizon=1000, debug_gui=False):
        manipulator_path = Path(path_robots).absolute().parent / 'data' / 'manipulator' / 'manipulator.urdf'
        self.robot_path = str(manipulator_path)
        print(self.robot_path)

        action_spec = [
            ("manipulator/joint_0", pybullet.VELOCITY_CONTROL),
            ("manipulator/finger_0_joint", pybullet.VELOCITY_CONTROL),
            ("manipulator/finger_1_joint", pybullet.VELOCITY_CONTROL),
            ("manipulator/finger_2_joint", pybullet.VELOCITY_CONTROL),
            ("manipulator/finger_3_joint", pybullet.VELOCITY_CONTROL)
        ]

        observation_spec = [
            ("manipulator/joint_0", PyBulletObservationType.JOINT_POS),
        ]

        files = {
            self.robot_path: dict(basePosition=[0.0, 0, 0.12],
                                  baseOrientation=[0, 0, 0.0, 1.0]),
            'plane.urdf': {}
        }

        super().__init__(files, action_spec, observation_spec, gamma, horizon, n_intermediate_steps=8, debug_gui=debug_gui,
                         distance=3, origin=[0., 0., 0.], angles=[0., -45., 0.])

        self._client.setGravity(0, 0, -9.81)

    def setup(self):
        # for i, (model_id, joint_id, _) in enumerate(self._action_data):
        #     self._client.resetJointState(model_id, joint_id, self.hexapod_initial_state[i])

        self._client.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=45.0, cameraPitch=-45,
                                                cameraTargetPosition=[0., 0., 0.])

        # for model_id, link_id in self._link_map.values():
        #     self._client.changeDynamics(model_id, link_id, lateralFriction=1.0, spinningFriction=1.0)
        #
        # for model_id in self._model_map.values():
        #     self._client.changeDynamics(model_id, -1, lateralFriction=1.0, spinningFriction=1.0)

    def reward(self, state, action, next_state):
        return 0.

    def is_absorbing(self, state):
        return False


if __name__ == '__main__':
    from mushroom_rl.core import Core, Agent
    from mushroom_rl.utils.dataset import compute_J


    class DummyAgent(Agent):
        def __init__(self, n_actions):
            self._n_actions = n_actions

        def draw_action(self, state):
            time.sleep(0.01)

            #return np.ones(self._n_actions)*0.1
            return np.random.randn(self._n_actions)

        def episode_start(self):
            pass

        def fit(self, dataset):
            pass


    mdp = ManipulatorBullet(debug_gui=True)
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
