import time
import numpy as np
import pybullet
import pybullet_data
from pathlib import Path
from mushroom_rl.environments.pybullet_envs.locomotion.locomotor_robot import LocomotorRobot


class HopperRobot(LocomotorRobot):
    def __init__(self, gamma=0.99, horizon=1000, debug_gui=False):
        hopper_path = Path(pybullet_data.getDataPath()) / 'mjcf' / 'hopper.xml'
        hopper_path = str(hopper_path)

        joints = ['thigh_joint', 'leg_joint', 'foot_joint']
        power = 0.75
        joint_power = np.array([100.0, 100.0, 100.0])

        super().__init__(hopper_path, joints, gamma, horizon, debug_gui, power, joint_power)

    def is_absorbing(self, state):
        pose = self._get_torso_pos(state)
        euler = pybullet.getEulerFromQuaternion(pose[3:])
        z = pose[2]
        pitch = euler[1]
        return z <= 0.8 or abs(pitch) >= 1.0


if __name__ == '__main__':
    from mushroom_rl.core import Core, Agent, Environment
    from mushroom_rl.utils.dataset import compute_J

    np.random.seed(1)

    def step_callback(dataset):
        step = dataset[0]
        action = step[1]
        reward = step[2]
        state = step[3]
        last = step[-1]

        print('action ', action)
        print('state ', state)
        print('reward ', reward)
        print(last)

    class DummyAgent(Agent):
        def __init__(self, n_actions):
            self._n_actions = n_actions

        def draw_action(self, state):
            time.sleep(1/60)
            #return 1e-3*np.ones(self._n_actions)
            return np.random.randn(self._n_actions)

        def episode_start(self):
            pass

        def fit(self, dataset):
            pass

    mdp = HopperRobot(debug_gui=True)
    # mdp = Environment.make('Gym.HopperBulletEnv-v0', render=True)

    agent = DummyAgent(mdp.info.action_space.shape[0])

    core = Core(agent, mdp, callback_step=step_callback)
    dataset = core.evaluate(n_episodes=10, render=False, quiet=True)
    print('reward: ', compute_J(dataset, mdp.info.gamma))
    print("mdp_info state shape", mdp.info.observation_space.shape)
    print("actual state shape", dataset[0][0].shape)
    print("mdp_info action shape", mdp.info.action_space.shape)
    print("actual action shape", dataset[0][1].shape)

    print("action low", mdp.info.action_space.low)
    print("action high", mdp.info.action_space.high)
