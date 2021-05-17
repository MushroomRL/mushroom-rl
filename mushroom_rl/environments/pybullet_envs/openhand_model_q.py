import time
import numpy as np
import pybullet
from mushroom_rl.environments.pybullet import PyBullet, PyBulletObservationType
from mushroom_rl.utils.spaces import Box

from pathlib import Path

from mushroom_rl.environments.pybullet_envs import __file__ as path_robots


class OpenHandModelQ(PyBullet):
    """
    The Yale OpenHand Model Q hand in a manipulation task.
    Open source hand available here:
    https://www.eng.yale.edu/grablab/openhand/

    This environment replicates in simulation the experiment described in:
    "Model Predictive Actor-Critic: Accelerating Robot Skill Acquisition with Deep Reinforcement Learning".
    Morgan A. S. et Al.. 2021.

    """
    def __init__(self, gamma=0.99, horizon=1000, valve_type='tri', debug_gui=False):
        """
        Constructor

        Args:
            gamma (float, 0.99): discount factor;
            horizon (int, 100): environment horizon;
            valve_type (str, 'tri'): type of valve to manipulate ('tri, 'quad', 'round');
            debug_gui (bool, False): flag to activate the debug visualizer.

        """
        assert valve_type in ['tri', 'quad', 'round']
        manipulator_path = Path(path_robots).absolute().parent / 'data' / 'openhand_model_q' / 'model_q.urdf'
        self.robot_path = str(manipulator_path)
        print(self.robot_path)

        action_spec = [
            ("base_rot_joint", pybullet.VELOCITY_CONTROL),
            # Left finger
            ("base_to_prox_l", pybullet.VELOCITY_CONTROL),
            ("prox_to_distal_l", pybullet.VELOCITY_CONTROL),
            # Right finger
            ("base_to_prox_r", pybullet.VELOCITY_CONTROL),
            ("prox_to_distal_r", pybullet.VELOCITY_CONTROL),
            # Connected finger left
            ("base_to_prox_cl", pybullet.VELOCITY_CONTROL),
            ("prox_to_distal_cl", pybullet.VELOCITY_CONTROL),
            # Connected finger right
            ("base_to_prox_cr", pybullet.VELOCITY_CONTROL),
            ("prox_to_distal_cr", pybullet.VELOCITY_CONTROL)
        ]

        observation_spec = [
            ("base_rot_joint", PyBulletObservationType.JOINT_POS),
            # Left finger
            ("base_to_prox_l", PyBulletObservationType.JOINT_POS),
            ("prox_to_distal_l", PyBulletObservationType.JOINT_POS),
            # Right finger
            ("base_to_prox_r", PyBulletObservationType.JOINT_POS),
            ("prox_to_distal_r", PyBulletObservationType.JOINT_POS),
            # Connected finger left
            ("base_to_prox_cl", PyBulletObservationType.JOINT_POS),
            ("prox_to_distal_cl", PyBulletObservationType.JOINT_POS),
            # Connected finger right
            ("base_to_prox_cr", PyBulletObservationType.JOINT_POS),
            ("prox_to_distal_cr", PyBulletObservationType.JOINT_POS),
            ("valve_joint", PyBulletObservationType.JOINT_VEL)
        ]

        valve_path = f'{valve_type}_valve.urdf'

        valve_path = Path(path_robots).absolute().parent / 'data' / 'openhand_model_q' / valve_path
        valve_path = str(valve_path)

        files = {
            self.robot_path: dict(basePosition=[0., 0., .225],
                                  baseOrientation=[0, 1, 0, 0],
                                  flags=pybullet.URDF_USE_SELF_COLLISION
                                  ),
            'plane.urdf': {},
            valve_path: dict(basePosition=[0., 0., 0.])
        }

        super().__init__(files, action_spec, observation_spec, gamma, horizon, n_intermediate_steps=8,
                         debug_gui=debug_gui, distance=0.5, origin=[0., 0., 0.2], angles=[0., -15., 0.])

        self._client.setGravity(0, 0, -9.81)

        self.gripper_initial_state = [0., 0., 0., 0., 0., 0., 0., 0., 0.]

    def _modify_mdp_info(self, mdp_info):

        low = mdp_info.action_space.low[:7]
        high = mdp_info.action_space.high[:7]
        reduced_action_space = Box(low=low, high=high)
        mdp_info.action_space = reduced_action_space

        return mdp_info

    def _compute_action(self, action):
        action_full = np.empty(9)

        action_full[:5] = action[:5]

        prox_c = action[5]
        distal_c = action[6]

        action_full[5] = prox_c
        action_full[6] = distal_c

        action_full[7] = prox_c
        action_full[8] = distal_c

        return action_full

    def setup(self):
        for i, (model_id, joint_id, _) in enumerate(self._action_data):
            self._client.resetJointState(model_id, joint_id, self.gripper_initial_state[i])

        self._client.resetDebugVisualizerCamera(cameraDistance=0.25, cameraYaw=45.0, cameraPitch=-15,
                                                cameraTargetPosition=[0., 0., 0.1])

    def reward(self, state, action, next_state):
        angular_velocity = self.get_sim_state(next_state, 'valve_joint', PyBulletObservationType.JOINT_VEL)

        return angular_velocity

    def is_absorbing(self, state):
        return False


if __name__ == '__main__':
    from mushroom_rl.core import Core, Agent
    from mushroom_rl.utils.dataset import compute_J


    class DummyAgent(Agent):
        def __init__(self, n_actions):
            self._n_actions = n_actions

        def draw_action(self, state):
            time.sleep(1/240)
            #return 0.1*np.ones(self._n_actions)
            return np.random.randn(self._n_actions)

        def episode_start(self):
            pass

        def fit(self, dataset):
            pass


    mdp = OpenHandModelQ(valve_type='tri', debug_gui=True)
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
