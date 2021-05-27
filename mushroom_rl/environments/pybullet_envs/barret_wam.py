import time
import numpy as np
import pybullet
from mushroom_rl.environments.pybullet import PyBullet, PyBulletObservationType

from pathlib import Path

from mushroom_rl.environments.pybullet_envs import __file__ as path_robots


class BatBot(PyBullet):
    def __init__(self, gamma=0.99, horizon=1000, debug_gui=False):
        robot_path = Path(path_robots).absolute().parent / 'data' / 'barret_wam' / 'wam_4dof.urdf'
        robot_path = str(robot_path)

        action_spec = [
            ("wam_j1_joint", pybullet.VELOCITY_CONTROL),
            ("wam_j2_joint", pybullet.VELOCITY_CONTROL),
            ("wam_j3_joint", pybullet.VELOCITY_CONTROL),
            ("wam_j4_joint", pybullet.VELOCITY_CONTROL)
        ]

        observation_spec = [
            ("wam_j1_joint", PyBulletObservationType.JOINT_POS),
            ("wam_j1_joint", PyBulletObservationType.JOINT_VEL),
            ("wam_j2_joint", PyBulletObservationType.JOINT_POS),
            ("wam_j2_joint", PyBulletObservationType.JOINT_VEL),
            ("wam_j3_joint", PyBulletObservationType.JOINT_POS),
            ("wam_j3_joint", PyBulletObservationType.JOINT_VEL),
            ("wam_j4_joint", PyBulletObservationType.JOINT_POS),
            ("wam_j4_joint", PyBulletObservationType.JOINT_VEL)
        ]

        files = {
            robot_path: dict(basePosition=[0.0, 0.0, 0.0],
                                  baseOrientation=[0, 0, 0.0, 1.0],
                                  flags=pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT),
            'plane.urdf': {}
        }

        super().__init__(files, action_spec, observation_spec, gamma, horizon, n_intermediate_steps=8, debug_gui=debug_gui,
                         distance=3, origin=[0., 0., 0.], angles=[0., -45., 0.])

        self._client.setGravity(0, 0, -9.81)

    def setup(self, state):
        self._client.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0.0, cameraPitch=-45,
                                                cameraTargetPosition=[0., 0., 0.])

    def reward(self, state, action, next_state, absorbing):
        return 0

    def is_absorbing(self, state):
        return False


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
            # return np.zeros(self._n_actions)

        def episode_start(self):
            pass

        def fit(self, dataset):
            pass


    mdp = BatBot(debug_gui=True)
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
