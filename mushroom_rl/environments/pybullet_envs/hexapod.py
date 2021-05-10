import time
import numpy as np
import pybullet
from mushroom_rl.environments.pybullet import PyBullet, PyBulletObservationType

from pathlib import Path

from mushroom_rl.environments.pybullet_envs import __file__ as path_robots


class HexapodBullet(PyBullet):
    def __init__(self, control_front=True, control_back=False, observe_opponent=False,
                 gamma=0.99, horizon=500, debug_gui=False):
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

            ("hexapod/body", PyBulletObservationType.LINK_POS),
            ("hexapod/body", PyBulletObservationType.LINK_LIN_VEL)
        ]

        files = {
            self.robot_path: dict(basePosition=[0.0, 0, 0.12],
                                  baseOrientation=[0, 0, 0.0, 1.0])
        }

        super().__init__(files, action_spec, observation_spec, gamma, horizon, debug_gui=debug_gui,
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

    def setup(self):
        for i, (model_id, joint_id, _) in enumerate(self._action_data):
            self._client.resetJointState(model_id, joint_id, self.hexapod_initial_state[i])

        self._client.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0.0, cameraPitch=-45,
                                                cameraTargetPosition=[0., 0., 0.])
    #     self.collision_filter()
    #
    # def collision_filter(self):
    #     # disable the collision with left and right rim Because of the inproper collision shape
    #     robot_links = ['F_link_1', 'F_link_2', 'F_link_3', 'F_link_striker_hand', 'F_link_striker_ee',
    #                    'B_link_1', 'B_link_2', 'B_link_3', 'B_link_striker_hand', 'B_link_striker_ee']
    #     table_rims = ['t_down_rim_l', 't_down_rim_r', 't_up_rim_l', 't_up_rim_l', 't_base']
    #     for link in robot_links:
    #         for table_r in table_rims:
    #             self._client.setCollisionFilterPair(self._link_map[link][0], self._link_map[table_r][0],
    #                                             self._link_map[link][1], self._link_map[table_r][1], 0)

    def reward(self, state, action, next_state):
        return 0.

    def is_absorbing(self, state):
        return False

    def _custom_load_models(self):

        plane = self._client.loadURDF('plane.urdf')

        return dict(plane=plane)

    @property
    def client(self):
        return self._client


if __name__ == '__main__':
    from mushroom_rl.core import Core
    from mushroom_rl.algorithms import Agent
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
 
