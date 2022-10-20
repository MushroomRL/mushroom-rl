import os

import numpy as np
import mujoco

from mushroom_rl.environments.mujoco import MuJoCo
from mushroom_rl.environments.mujoco import ObservationType

from mushroom_rl.environments.mujoco_envs import __file__ as path_robots


class AirHockeyBase(MuJoCo):
    def __init__(self, gamma=0.99, horizon=500, n_agents=1, env_noise=False, obs_noise=False, torque_control=True,
                 step_action_function=None, timestep=1 / 240., n_intermediate_steps=1):

        self.n_agents = n_agents
        self.env_noise = env_noise
        self.obs_noise = obs_noise
        self.step_action_function = step_action_function

        self.agents = []

        action_spec = []
        observation_spec = [("puck_pos", "puck", ObservationType.BODY_POS),
                            ("puck_vel", "puck", ObservationType.BODY_VEL)]
        additional_data = []
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
                scene = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "data", "air_hockey",
                                          "double.xml")

                action_spec += ["planar_robot_2/joint_1", "planar_robot_2/joint_2", "planar_robot_2/joint_3"]
                # Add puck pos/vel again to transform into second agents frame
                observation_spec += [("robot_2/puck_pos", "puck", ObservationType.BODY_POS),
                                     ("robot_2/puck_vel", "puck", ObservationType.BODY_VEL),
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

        super(AirHockeyBase, self).__init__(scene, action_spec, observation_spec, gamma, horizon, timestep,
                                            1, n_intermediate_steps, additional_data, collision_spec)

        # URDF fot pinoccio
        robot_urdf = os.path.join(os.path.dirname(os.path.abspath(path_robots)), "data", "air_hockey",
                                  "planar_robot.urdf")

        self.env_spec = dict()
        self.env_spec['table'] = {"length": 1.96, "width": 1.02, "height": -0.189, "goal": 0.25}
        self.env_spec['puck'] = {"radius": 0.03165}
        self.env_spec['mallet'] = {"radius": 0.05}

        agent_spec = dict()
        agent_spec['name'] = "planar_robot_1"
        agent_spec.update({'link_length': [0.5, 0.4, 0.4]})
        agent_spec["urdf"] = robot_urdf

        agent_spec['frame'] = np.eye(4)
        temp = np.zeros((9, 1))
        mujoco.mju_quat2Mat(temp, self._model.body("planar_robot_1/base").quat)
        agent_spec['frame'][:3, :3] = temp.reshape(3, 3)
        agent_spec['frame'][:3, 3] = self._model.body("planar_robot_1/base").pos
        self.agents.append(agent_spec)

        if self.n_agents == 2:
            agent_spec = dict()
            agent_spec['name'] = "planar_robot_2"
            agent_spec.update({'link_length': [0.5, 0.4, 0.4]})
            agent_spec["urdf"] = robot_urdf

            agent_spec['frame'] = np.eye(4)
            temp = np.zeros((9, 1))
            mujoco.mju_quat2Mat(temp, self._model.body("planar_robot_2/base").quat)
            agent_spec['frame'][:3, :3] = temp.reshape(3, 3)
            agent_spec['frame'][:3, 3] = self._model.body("planar_robot_2/base").pos
            self.agents.append(agent_spec)

    def _compute_action(self, obs, action):
        if self.step_action_function is None:
            return action
        else:
            return self.step_action_function(obs, action)

    def _simulation_pre_step(self):
        if self.env_noise:
            force = np.random.randn(2) * 0.0005
            self._data.body("puck").xfrc_applied[:2] = force

    def is_absorbing(self, obs):
        boundary = np.array([self.env_spec['table']['length'], self.env_spec['table']['width']]) / 2
        puck_pos = self.obs_helper.get_from_obs(obs, "puck_pos")

        if np.any(np.abs(puck_pos[:2]) > boundary):
            return True
        return False

    def _puck_2d_in_robot_frame(self, puck_in, robot_frame, type='pose'):
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
