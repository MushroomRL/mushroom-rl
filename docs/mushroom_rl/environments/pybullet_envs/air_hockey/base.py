import os
import time

import numpy as np
import pybullet
import pybullet_utils.transformations as transformations

from mushroom_rl.environments.pybullet import PyBullet, PyBulletObservationType
from mushroom_rl.environments.pybullet_envs import __file__ as path_robots


class AirHockeyBaseBullet(PyBullet):
    """
    Base class for air hockey environment.
    The environment is designed for 3 joints planar robot playing Air-Hockey
    """
    def __init__(self, gamma=0.99, horizon=500, n_agents=1, env_noise=False, obs_noise=False, obs_delay=False,
                 torque_control=True, step_action_function=None, timestep=1 / 240., n_intermediate_steps=1,
                 debug_gui=False, table_boundary_terminate=False):
        """
        Constructor.

        Args:
            gamma (float, 0.99): discount factor;
            horizon (int, 500): horizon of the task;
            n_agents (int, 1): number of agents;
            env_noise(bool, False): If true, the puck's movement is affected by the air-flow noise;
            obs_noise(bool, False): If true, the noise is added in the observation;
            obs_delay(bool, False): If true, velocity is observed by the low-pass filter;
            control(bool, True): If false, the robot in position control mode;
            step_action_function(object, None): A callable function to warp-up the policy action to environment command.
            table_boundary_terminate(bool, False): Episode terminates if the mallet is outside the boundary
        """

        self.n_agents = n_agents
        self.env_noise = env_noise
        self.obs_noise = obs_noise
        self.obs_delay = obs_delay
        self.step_action_function = step_action_function
        self.table_boundary_terminate = table_boundary_terminate

        puck_file = os.path.join(os.path.dirname(os.path.abspath(path_robots)),
                                 "data", "air_hockey", "puck.urdf")
        table_file = os.path.join(os.path.dirname(os.path.abspath(path_robots)),
                                  "data", "air_hockey", "air_hockey_table.urdf")
        robot_file_1 = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "data",
                                    "air_hockey", "planar", "planar_robot_1.urdf")
        robot_file_2 = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "data",
                                    "air_hockey", "planar", "planar_robot_2.urdf")

        model_files = dict()
        model_files[puck_file] = dict(flags=pybullet.URDF_USE_IMPLICIT_CYLINDER,
                                      basePosition=[0.0, 0, -0.189], baseOrientation=[0, 0, 0.0, 1.0])
        model_files[table_file] = dict(useFixedBase=True, basePosition=[0.0, 0, -0.189],
                                       baseOrientation=[0, 0, 0.0, 1.0])

        actuation_spec = list()
        observation_spec = [("puck", PyBulletObservationType.BODY_POS),
                            ("puck", PyBulletObservationType.BODY_LIN_VEL),
                            ("puck", PyBulletObservationType.BODY_ANG_VEL)]
        self.agents = []

        if torque_control:
            control = pybullet.TORQUE_CONTROL
        else:
            control = pybullet.POSITION_CONTROL

        if 1 <= self.n_agents <= 2:
            agent_spec = dict()
            agent_spec['name'] = "planar_robot_1"
            agent_spec.update({'link_length': [0.5, 0.4, 0.4], "urdf": robot_file_1})
            translate = [-1.51, 0, 0.0]
            quaternion = [0.0, 0.0, 0.0, 1.0]
            agent_spec['frame'] = transformations.translation_matrix(translate)
            agent_spec['frame'] = agent_spec['frame'] @ transformations.quaternion_matrix(quaternion)
            model_files[robot_file_1] = dict(
                flags=pybullet.URDF_USE_IMPLICIT_CYLINDER | pybullet.URDF_USE_INERTIA_FROM_FILE,
                basePosition=translate, baseOrientation=quaternion)

            self.agents.append(agent_spec)
            actuation_spec += [("planar_robot_1/joint_1", control),
                               ("planar_robot_1/joint_2", control),
                               ("planar_robot_1/joint_3", control)]
            observation_spec += [("planar_robot_1/joint_1", PyBulletObservationType.JOINT_POS),
                                 ("planar_robot_1/joint_2", PyBulletObservationType.JOINT_POS),
                                 ("planar_robot_1/joint_3", PyBulletObservationType.JOINT_POS),
                                 ("planar_robot_1/joint_1", PyBulletObservationType.JOINT_VEL),
                                 ("planar_robot_1/joint_2", PyBulletObservationType.JOINT_VEL),
                                 ("planar_robot_1/joint_3", PyBulletObservationType.JOINT_VEL),
                                 ("planar_robot_1/link_striker_ee", PyBulletObservationType.LINK_POS),
                                 ("planar_robot_1/link_striker_ee", PyBulletObservationType.LINK_LIN_VEL)]

            if self.n_agents == 2:
                agent_spec = dict()
                agent_spec['name'] = "planar_robot_2"
                agent_spec.update({'link_length': [0.5, 0.4, 0.4], "urdf": robot_file_1})
                translate = [1.51, 0, 0.0]
                quaternion = [0.0, 0.0, 1.0, 0.0]
                agent_spec['frame'] = transformations.translation_matrix(translate)
                agent_spec['frame'] = agent_spec['frame'] @ transformations.quaternion_matrix(quaternion)
                model_files[robot_file_2] = dict(
                    flags=pybullet.URDF_USE_IMPLICIT_CYLINDER | pybullet.URDF_USE_INERTIA_FROM_FILE,
                    basePosition=translate, baseOrientation=quaternion)
                self.agents.append(agent_spec)

                actuation_spec += [("planar_robot_2/joint_1", control),
                                   ("planar_robot_2/joint_2", control),
                                   ("planar_robot_2/joint_3", control)]
                observation_spec += [("planar_robot_2/joint_1", PyBulletObservationType.JOINT_POS),
                                     ("planar_robot_2/joint_2", PyBulletObservationType.JOINT_POS),
                                     ("planar_robot_2/joint_3", PyBulletObservationType.JOINT_POS),
                                     ("planar_robot_2/joint_1", PyBulletObservationType.JOINT_VEL),
                                     ("planar_robot_2/joint_2", PyBulletObservationType.JOINT_VEL),
                                     ("planar_robot_2/joint_3", PyBulletObservationType.JOINT_VEL),
                                     ("planar_robot_2/link_striker_ee", PyBulletObservationType.LINK_POS),
                                     ("planar_robot_2/link_striker_ee", PyBulletObservationType.LINK_LIN_VEL)]
        else:
            raise ValueError('n_agents should be 1 or 2')

        super().__init__(model_files, actuation_spec, observation_spec, gamma,
                         horizon, timestep=timestep, n_intermediate_steps=n_intermediate_steps,
                         debug_gui=debug_gui, size=(500, 500), distance=1.8)
        self._client.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=0.0, cameraPitch=-89.9,
                                                cameraTargetPosition=[0., 0., 0.])
        self.env_spec = dict()
        self.env_spec['table'] = {"length": 1.96, "width": 1.02, "height": -0.189, "goal": 0.25, "urdf": table_file}
        self.env_spec['puck'] = {"radius": 0.03165, "urdf": puck_file}
        self.env_spec['mallet'] = {"radius": 0.05}
        self.env_spec['joint_vel_threshold'] = 0.1

    def _compute_action(self, state, action):
        if self.step_action_function is None:
            return action
        else:
            return self.step_action_function(state, action)

    def _simulation_pre_step(self):
        if self.env_noise:
            force = np.concatenate([np.random.randn(2), [0]]) * 0.0005
            self._client.applyExternalForce(self._model_map['puck']['id'], -1, force, [0., 0., 0.],
                                            self._client.WORLD_FRAME)

    def is_absorbing(self, state):
        boundary = np.array([self.env_spec['table']['length'], self.env_spec['table']['width']]) / 2
        puck_pos = self.get_sim_state(state, "puck", PyBulletObservationType.BODY_POS)[:3]
        if np.any(np.abs(puck_pos[:2]) > boundary) or abs(puck_pos[2] - self.env_spec['table']['height']) > 0.05:
            return True

        if self.table_boundary_terminate:
            if self.n_agents >= 1:
                ee_pos = self.get_sim_state(state, "planar_robot_1/link_striker_ee",
                                            PyBulletObservationType.LINK_POS)[:3]
                if abs(ee_pos[0]) > self.env_spec['table']['length'] / 2 or \
                        abs(ee_pos[1]) > self.env_spec['table']['width'] / 2:
                    return True
            if self.n_agents == 2:
                ee_pos = self.get_sim_state(state, "planar_robot_2/link_striker_ee",
                                            PyBulletObservationType.LINK_POS)[:3]
                if abs(ee_pos[0]) > self.env_spec['table']['length'] / 2 or \
                        abs(ee_pos[1]) > self.env_spec['table']['width'] / 2:
                    return True
        return False

    def forward_kinematics(self, joint_state):
        x = np.cos(joint_state[0]) * self.agents[0]['link_length'][0] + \
            np.cos(joint_state[0] + joint_state[1]) * self.agents[0]['link_length'][1] + \
            np.cos(joint_state[0] + joint_state[1] + joint_state[2]) * self.agents[0]['link_length'][2]

        y = np.sin(joint_state[0]) * self.agents[0]['link_length'][0] + \
            np.sin(joint_state[0] + joint_state[1]) * self.agents[0]['link_length'][1] + \
            np.sin(joint_state[0] + joint_state[1] + joint_state[2]) * self.agents[0]['link_length'][2]

        yaw = joint_state[0] + joint_state[1] + joint_state[2]

        return np.array([x, y, yaw])
