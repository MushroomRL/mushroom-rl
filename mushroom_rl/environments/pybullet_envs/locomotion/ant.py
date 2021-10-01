import time
import numpy as np
import pybullet_data
from pathlib import Path
from mushroom_rl.environments.pybullet_envs.locomotion.locomotor_robot import LocomotorRobot


class AntRobot(LocomotorRobot):
    def __init__(self, gamma=0.99, horizon=1000, debug_gui=False):
        ant_path = Path(pybullet_data.getDataPath()) / 'mjcf' / 'ant.xml'
        ant_path = str(ant_path)

        joints = ['hip_1', 'ankle_1', 'hip_2', 'ankle_2', 'hip_3', 'ankle_3', 'hip_4', 'ankle_4']
        contacts = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']
        power = 2.5
        joint_power = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])

        super().__init__('ant', ant_path, joints, contacts, gamma, horizon, debug_gui, power, joint_power,
                         bidimensional=False)

    def is_absorbing(self, state):
        pose = self._get_torso_pos(state)
        z = pose[2]

        return z <= 0.26


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

    mdp = AntRobot(debug_gui=True)
    # mdp = Environment.make('Gym.HopperBulletEnv-v0', render=True)

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
