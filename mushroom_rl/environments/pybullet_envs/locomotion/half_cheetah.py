import time
import numpy as np
import pybullet
import pybullet_data
from pathlib import Path
from mushroom_rl.environments.pybullet_envs.locomotion.locomotor_robot import BidimensionalLocomotorRobot


class HalfCheetahRobot(BidimensionalLocomotorRobot):
    def __init__(self, gamma=0.99, horizon=1000, debug_gui=False):
        cheetah_path = Path(pybullet_data.getDataPath()) / 'mjcf' / 'half_cheetah.xml'
        cheetah_path = str(cheetah_path)

        joints = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
        contacts = ['bthigh', 'bshin', 'fthigh', 'fshin', 'bfoot', 'ffoot']
        power = 0.9
        joint_power = np.array([120.0, 90.0, 60.0, 140.0, 60.0, 30.0])

        super().__init__('cheetah', cheetah_path, joints, contacts, gamma, horizon, debug_gui, power, joint_power)

    def is_absorbing(self, state):
        contacts = state[-6:-2]
        if np.any(contacts == 1):
            return True

        pose = self._get_torso_pos(state)
        euler = pybullet.getEulerFromQuaternion(pose[3:])
        pitch = euler[1]

        return np.abs(pitch) >= 1.0


if __name__ == '__main__':
    from mushroom_rl.core import Core, Agent, Environment
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

    mdp = HalfCheetahRobot(debug_gui=True)

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
