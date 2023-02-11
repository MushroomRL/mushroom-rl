import os

import numpy as np
import mujoco

from mushroom_rl.environments.mujoco import MuJoCo
from mushroom_rl.environments.mujoco import ObservationType

from mushroom_rl.environments.mujoco_envs import __file__ as path_robots


class AirHockeyBase(MuJoCo):
    """
    Abstract class for all AirHockey Environments.

    """
    def __init__(self, n_agents=1, env_noise=False, obs_noise=False, gamma=0.99, horizon=500,
                 timestep=1 / 240., n_substeps=1, n_intermediate_steps=1, **viewer_params):
        """
        Constructor.

        Args:
            n_agents (int, 1): number of agent to be used in the environment (one or two)
            env_noise (bool, False): if True, the environment uses noisy dynamics.
            obs_noise (bool, False): if True, the environment uses noisy observations.

        """

        self.n_agents = n_agents
        self.env_noise = env_noise
        self.obs_noise = obs_noise

        self.agents = []

        action_spec = []
        observation_spec = [("puck_pos", "puck", ObservationType.JOINT_POS),
                            ("puck_vel", "puck", ObservationType.JOINT_VEL)]

        additional_data = [("puck_pos", "puck", ObservationType.JOINT_POS),
                           ("puck_vel", "puck", ObservationType.JOINT_VEL)]

        collision_spec = [("puck", ["puck"]),
                          ("rim", ["rim_home_l", "rim_home_r", "rim_away_l", "rim_away_l", "rim_left", "rim_right"]),
                          ("rim_short_sides", ["rim_home_l", "rim_home_r", "rim_away_l", "rim_away_r"])]

        if 1 <= self.n_agents <= 2:
            scene = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "data", "air_hockey",
                                      "single.xml")

            action_spec += ["planar_robot_1/joint_1", "planar_robot_1/joint_2", "planar_robot_1/joint_3"]
            observation_spec += [("robot_1/joint_1_pos", "planar_robot_1/joint_1", ObservationType.JOINT_POS),
                                 ("robot_1/joint_2_pos", "planar_robot_1/joint_2", ObservationType.JOINT_POS),
                                 ("robot_1/joint_3_pos", "planar_robot_1/joint_3", ObservationType.JOINT_POS),
                                 ("robot_1/joint_1_vel", "planar_robot_1/joint_1", ObservationType.JOINT_VEL),
                                 ("robot_1/joint_2_vel", "planar_robot_1/joint_2", ObservationType.JOINT_VEL),
                                 ("robot_1/joint_3_vel", "planar_robot_1/joint_3", ObservationType.JOINT_VEL)]

            additional_data += [("robot_1/ee_pos", "planar_robot_1/body_ee", ObservationType.BODY_POS),
                                ("robot_1/ee_vel", "planar_robot_1/body_ee", ObservationType.BODY_VEL)]

            collision_spec += [("robot_1/ee", ["planar_robot_1/ee"])]

            if self.n_agents == 2:
                scene = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "data", "air_hockey", "double.xml")

                action_spec += ["planar_robot_2/joint_1", "planar_robot_2/joint_2", "planar_robot_2/joint_3"]
                # Add puck pos/vel again to transform into second agents frame
                observation_spec += [("robot_2/puck_pos", "puck", ObservationType.JOINT_POS),
                                     ("robot_2/puck_vel", "puck", ObservationType.JOINT_VEL),
                                     ("robot_2/joint_1_pos", "planar_robot_2/joint_1", ObservationType.JOINT_POS),
                                     ("robot_2/joint_2_pos", "planar_robot_2/joint_2", ObservationType.JOINT_POS),
                                     ("robot_2/joint_3_pos", "planar_robot_2/joint_3", ObservationType.JOINT_POS),
                                     ("robot_2/joint_1_vel", "planar_robot_2/joint_1", ObservationType.JOINT_VEL),
                                     ("robot_2/joint_2_vel", "planar_robot_2/joint_2", ObservationType.JOINT_VEL),
                                     ("robot_2/joint_3_vel", "planar_robot_2/joint_3", ObservationType.JOINT_VEL)]

                additional_data += [("robot_2/ee_pos", "planar_robot_2/body_ee", ObservationType.BODY_POS),
                                    ("robot_2/ee_vel", "planar_robot_2/body_ee", ObservationType.BODY_VEL)]

                collision_spec += [("robot_2/ee", ["planar_robot_2/ee"])]
        else:
            raise ValueError('n_agents should be 1 or 2')


        self.env_spec = dict()
        self.env_spec['table'] = {"length": 1.96, "width": 1.02, "height": -0.189, "goal": 0.25}
        self.env_spec['puck'] = {"radius": 0.03165}
        self.env_spec['mallet'] = {"radius": 0.05}

        # Load just the robot model for to use for kinematics
        robot_path = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "data", "air_hockey", "planar_robot_1.xml")

        self.robot_model = mujoco.MjModel.from_xml_path(robot_path)
        # Move robot to zero pos
        self.robot_model.body("planar_robot_1/base").pos = np.zeros(3)
        self.robot_data = mujoco.MjData(self.robot_model)

        # Properties of the robots
        agent_spec = dict()
        agent_spec['name'] = "planar_robot_1"
        agent_spec.update({'link_length': [0.5, 0.4, 0.4],
                           'max_joint_vel': np.array([np.pi / 2, np.pi / 2, np.pi * 2 / 3])})

        self.agents.append(agent_spec)
        if self.n_agents == 2:
            agent_spec = dict()
            agent_spec['name'] = "planar_robot_2"
            agent_spec.update({'link_length': [0.5, 0.4, 0.4],
                               'max_joint_vel': np.array([np.pi / 2, np.pi / 2, np.pi * 2 / 3])})

            self.agents.append(agent_spec)

        max_joint_vel = np.concatenate([spec['max_joint_vel'] for spec in self.agents])
        super().__init__(scene, action_spec, observation_spec, gamma, horizon, timestep, n_substeps,
                         n_intermediate_steps, additional_data, collision_spec, max_joint_vel,
                         **viewer_params)

        # Get the transformations from table to robot coordinate system
        for i, agent_spec in enumerate(self.agents):
            agent_spec['frame'] = np.eye(4)
            temp = np.zeros((9, 1))
            mujoco.mju_quat2Mat(temp, self._model.body("planar_robot_" + str(i+1) + "/base").quat)
            agent_spec['frame'][:3, :3] = temp.reshape(3, 3)
            agent_spec['frame'][:3, 3] = self._model.body("planar_robot_" + str(i+1) + "/base").pos

    def _simulation_pre_step(self):
        if self.env_noise:
            force = np.random.randn(2) * 0.0005
            self._data.body("puck").xfrc_applied[:2] = force

    def is_absorbing(self, obs):
        boundary = np.array([self.env_spec['table']['length'], self.env_spec['table']['width']]) / 2
        puck_pos = self.obs_helper.get_from_obs(obs, "puck_pos")
        puck_vel = self.obs_helper.get_from_obs(obs, "puck_vel")

        if np.any(np.abs(puck_pos[:2]) > boundary) or np.linalg.norm(puck_vel) > 100:
            return True
        return False

    @staticmethod
    def _puck_2d_in_robot_frame(puck_in, robot_frame, type='pose'):
        if type == 'pose':
            puck_frame = np.eye(4)
            puck_frame[:2, 3] = puck_in

            frame_target = np.linalg.inv(robot_frame) @ puck_frame
            puck_in[:] = frame_target[:2, 3]

        if type == 'vel':
            rot_mat = robot_frame[:3, :3]

            vel_lin = np.zeros(3)
            vel_lin[:2] = puck_in[1:]

            vel_target = rot_mat.T @ vel_lin

            puck_in[1:] = vel_target[:2]
