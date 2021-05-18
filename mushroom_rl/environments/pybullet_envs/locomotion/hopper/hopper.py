import time
import numpy as np
import pybullet
from mushroom_rl.environments.pybullet import PyBulletObservationType
from mushroom_rl.environments.pybullet_envs.locomotion.locomotor_robot import LocomotorRobot

from pathlib import Path

from mushroom_rl.environments.pybullet_envs import __file__ as path_robots


class HopperRobot(LocomotorRobot):
    def __init__(self, gamma=0.99, horizon=1000, debug_gui=False):
        hopper_path = Path(path_robots).absolute().parent / 'data' / 'hopper' / 'hopper.xml'
        hopper_path = str(hopper_path)

        action_spec = [
            ("thigh_joint", pybullet.TORQUE_CONTROL),
            ("leg_joint", pybullet.TORQUE_CONTROL),
            ("foot_joint", pybullet.TORQUE_CONTROL),
        ]

        observation_spec = [
            ("thigh_joint", PyBulletObservationType.JOINT_POS),
            ("thigh_joint", PyBulletObservationType.JOINT_VEL),
            ("leg_joint", PyBulletObservationType.JOINT_POS),
            ("leg_joint", PyBulletObservationType.JOINT_VEL),
            ("foot_joint", PyBulletObservationType.JOINT_POS),
            ("foot_joint", PyBulletObservationType.JOINT_VEL),
        ]

        super().__init__(hopper_path, action_spec, observation_spec, gamma, horizon, debug_gui)

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
            time.sleep(1/60)

            return np.random.randn(self._n_actions)

        def episode_start(self):
            pass

        def fit(self, dataset):
            pass


    mdp = HopperRobot(debug_gui=True)
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
