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
    def __init__(self, gamma=0.99, horizon=1000, object_type='tri', debug_gui=False):
        """
        Constructor

        Args:
            gamma (float, 0.99): discount factor;
            horizon (int, 100): environment horizon;
            object_type (str, 'tri'): type of object to manipulate ('tri, 'quad', 'round', '');
            debug_gui (bool, False): flag to activate the debug visualizer.

        """
        assert object_type in ['tri', 'quad', 'round', 'apple']
        manipulator_path = Path(path_robots).absolute().parent / 'data' / 'openhand_model_q' / 'model_q.urdf'
        robot_path = str(manipulator_path)
        self._finger_gating = object_type == 'apple'

        action_spec = [
            ("base_rot_joint", pybullet.TORQUE_CONTROL),
            # Left finger
            ("base_to_prox_l", pybullet.TORQUE_CONTROL),
            ("prox_to_distal_l", pybullet.TORQUE_CONTROL),
            # Right finger
            ("base_to_prox_r", pybullet.TORQUE_CONTROL),
            ("prox_to_distal_r", pybullet.TORQUE_CONTROL),
            # Connected finger left
            ("base_to_prox_cl", pybullet.TORQUE_CONTROL),
            ("prox_to_distal_cl", pybullet.TORQUE_CONTROL),
            # Connected finger right
            ("base_to_prox_cr", pybullet.TORQUE_CONTROL),
            ("prox_to_distal_cr", pybullet.TORQUE_CONTROL)
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
            ("prox_to_distal_cr", PyBulletObservationType.JOINT_POS)
        ]

        if self._finger_gating:
            observation_spec += [
                ("apple", PyBulletObservationType.BODY_POS),
                ("apple", PyBulletObservationType.BODY_ANG_VEL)
            ]
        else:
            observation_spec += [
                ("valve_joint", PyBulletObservationType.JOINT_VEL)
            ]

        files = dict()

        files[robot_path] = dict(basePosition=[0., 0., .225],
                                 baseOrientation=[0, 1, 0, 0],
                                 flags=pybullet.URDF_USE_SELF_COLLISION | pybullet.URDF_USE_INERTIA_FROM_FILE)
        files['plane.urdf'] = dict()

        if object_type != 'apple':
            object_path = f'{object_type}_valve.urdf'
            object_dict = dict(basePosition=[0., 0., 0.])
        else:
            object_path = 'apple.sdf'
            object_dict = dict()

        object_path = Path(path_robots).absolute().parent / 'data' / 'openhand_model_q' / object_path
        files[str(object_path)] = object_dict

        super().__init__(files, action_spec, observation_spec, gamma, horizon, timestep=1/960, n_intermediate_steps=8,
                         debug_gui=debug_gui, distance=0.5, origin=[0., 0., 0.2], angles=[0., -15., 0.])

        self._load_texture()

        self._client.setGravity(0, 0, -9.81)

        if self._finger_gating:
            self.gripper_initial_state = [0., np.pi/3, np.pi/6, np.pi/3, np.pi/6, np.pi/3, np.pi/6, np.pi/3, np.pi/6]
        else:
            self.gripper_initial_state = [0., 0., 0., 0., 0., 0., 0., 0., 0.]

        self.apple_initial_position = [0., 0., 0.07]

        r_actuator = 1.39e-2
        r_proximal = 1e-2
        r_distal = 8e-3

        self._R = np.array(
            [
                [r_proximal, 0., 0.],
                [r_distal, 0., 0.],
                [0., r_proximal, 0.],
                [0., r_distal, 0.],
                [0., 0., r_proximal],
                [0., 0., r_distal],
                [0., 0., r_proximal],
                [0., 0., r_distal],
            ]
        ).T

        e_proximal = 6.25
        e_distal = 17.86

        self._E = np.diag([e_proximal, e_distal]*4)

    def _modify_mdp_info(self, mdp_info):
        rot_joint_low = mdp_info.action_space.low[0]
        rot_joint_high = mdp_info.action_space.high[0]

        low = np.array([rot_joint_low, 0., 0., 0.])
        high = np.array([rot_joint_high, 10., 10., 20.])
        reduced_action_space = Box(low=low, high=high)
        mdp_info.action_space = reduced_action_space

        return mdp_info

    def _compute_action(self, state, action):
        action_full = np.empty(9)

        action_full[0] = action[0]

        f = action[1:]
        q = self.joints.positions(state)[1:]

        action_full[1:] = self._R.T @ f - self._E @ q

        print(action_full)
        return action_full

    def setup(self, state):
        for i, (model_id, joint_id, _) in enumerate(self._indexer.action_data):
            self._client.resetJointState(model_id, joint_id, self.gripper_initial_state[i])

        self._client.resetDebugVisualizerCamera(cameraDistance=0.25, cameraYaw=45.0, cameraPitch=-15,
                                                cameraTargetPosition=[0., 0., 0.1])

        if self._finger_gating:
            _, orientation = self._client.getBasePositionAndOrientation(self._model_map['apple'])
            self._client.resetBasePositionAndOrientation(self._model_map['apple'], self.apple_initial_position,
                                                         orientation)
            self._client.changeDynamics(self._model_map['apple'], -1,
                                        restitution=0., contactStiffness=1e+09, contactDamping=0.9)

    def reward(self, state, action, next_state, absorbing):
        if self._finger_gating:
            angular_velocity = self.get_sim_state(next_state, 'apple', PyBulletObservationType.BODY_ANG_VEL)[2]
        else:
            angular_velocity = self.get_sim_state(next_state, 'valve_joint', PyBulletObservationType.JOINT_VEL)

        return angular_velocity

    def is_absorbing(self, state):
        if self._finger_gating:
            collisions = self._client.getContactPoints(self._model_map['plane'], self._model_map['apple'])

            return len(collisions) > 0

        else:
            return False

    def _load_texture(self):
        if self._finger_gating:
            texture_path = Path(path_robots).absolute().parent / 'data' / 'openhand_model_q' /\
                           'meshes' / 'apple' / 'texture_map.png'
            texUid = self._client.loadTexture(str(texture_path))
            self._client.changeVisualShape(self._model_map['apple'], -1, textureUniqueId=texUid)


if __name__ == '__main__':
    from mushroom_rl.core import Core, Agent
    from mushroom_rl.utils.dataset import compute_J


    class DummyAgent(Agent):
        def __init__(self, n_actions, dt):
            self._n_actions = n_actions
            self.dt = dt

        def draw_action(self, state):
            time.sleep(self.dt)

            action = 10*np.ones(self._n_actions)
            #action[0] = 1e-6
            #action[-1] = 0
            # action = np.random.randn(self._n_actions)
            #action = np.zeros(self._n_actions)
            return action

        def episode_start(self):
            pass

        def fit(self, dataset):
            pass


    mdp = OpenHandModelQ(object_type='apple', debug_gui=True)
    agent = DummyAgent(mdp.info.action_space.shape[0], mdp.dt)

    core = Core(agent, mdp)
    dataset = core.evaluate(n_episodes=10, render=False)
    print('reward: ', compute_J(dataset, mdp.info.gamma))
    print("mdp_info state shape", mdp.info.observation_space.shape)
    print("actual state shape", dataset[0][0].shape)
    print("mdp_info action shape", mdp.info.action_space.shape)
    print("actual action shape", dataset[0][1].shape)

    print("action low", mdp.info.action_space.low)
    print("action high", mdp.info.action_space.high)
