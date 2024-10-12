from mushroom_rl.environments.mujoco import MuJoCo, ObservationType
import numpy as np
from pathlib import Path
from copy import deepcopy

import mujoco


class BallInACup(MuJoCo):
    """
    Mujoco simulation of Ball In A Cup task, using Barret WAM robot.

    """
    def __init__(self):
        """
        Constructor.

        """
        xml_path = (Path(__file__).resolve().parent / "data" / "ball_in_a_cup" / "model.xml").as_posix()
        action_spec = ["act/wam/base_yaw_joint", "act/wam/shoulder_pitch_joint", "act/wam/shoulder_yaw_joint",
                       "act/wam/elbow_pitch_joint", "act/wam/wrist_yaw_joint", "act/wam/wrist_pitch_joint",
                       "act/wam/palm_yaw_joint"]

        observation_spec = [("base_yaw_pos", "wam/base_yaw_joint", ObservationType.JOINT_POS),
                            ("base_yaw_vel", "wam/base_yaw_joint", ObservationType.JOINT_VEL),
                            ("shoulder_pitch_pos", "wam/shoulder_pitch_joint", ObservationType.JOINT_POS),
                            ("shoulder_pitch_vel", "wam/shoulder_pitch_joint", ObservationType.JOINT_VEL),
                            ("shoulder_yaw_pos", "wam/shoulder_yaw_joint", ObservationType.JOINT_POS),
                            ("shoulder_yaw_vel", "wam/shoulder_yaw_joint", ObservationType.JOINT_VEL),
                            ("elbow_pitch_pos", "wam/elbow_pitch_joint", ObservationType.JOINT_POS),
                            ("elbow_pitch_vel", "wam/elbow_pitch_joint", ObservationType.JOINT_VEL),
                            ("wrist_yaw_pos", "wam/wrist_yaw_joint", ObservationType.JOINT_POS),
                            ("wrist_yaw_vel", "wam/wrist_yaw_joint", ObservationType.JOINT_VEL),
                            ("wrist_pitch_pos", "wam/wrist_pitch_joint", ObservationType.JOINT_POS),
                            ("wrist_pitch_vel", "wam/wrist_pitch_joint", ObservationType.JOINT_VEL),
                            ("palm_yaw_pos", "wam/palm_yaw_joint", ObservationType.JOINT_POS),
                            ("palm_yaw_vel", "wam/palm_yaw_joint", ObservationType.JOINT_VEL),
                            ("ball_pos", "ball", ObservationType.BODY_POS),
                            ("ball_vel", "ball", ObservationType.BODY_VEL_WORLD)]

        additional_data_spec = [("ball_pos", "ball", ObservationType.BODY_POS),
                                ("goal_pos", "cup_goal_final", ObservationType.SITE_POS)]

        collision_groups = [("ball", ["ball_geom"]),
                            ("robot", ["cup_geom1", "cup_geom2", "wrist_palm_link_convex_geom",
                                       "wrist_pitch_link_convex_decomposition_p1_geom",
                                       "wrist_pitch_link_convex_decomposition_p2_geom",
                                       "wrist_pitch_link_convex_decomposition_p3_geom",
                                       "wrist_yaw_link_convex_decomposition_p1_geom",
                                       "wrist_yaw_link_convex_decomposition_p2_geom",
                                       "forearm_link_convex_decomposition_p1_geom",
                                       "forearm_link_convex_decomposition_p2_geom"])]

        super().__init__(xml_path, action_spec, observation_spec, 0.9999, 2000, n_intermediate_steps=4,
                         additional_data_spec=additional_data_spec, collision_groups=collision_groups)

        self.init_robot_pos = np.array([0.0, 0.58760536, 0.0, 1.36004913, 0.0, -0.32072943, -1.57])
        self.p_gains = np.array([200, 400, 100, 100, 10, 10, 5])
        self.d_gains = np.array([7, 15, 5, 2.5, 0.3, 0.3, 0.05])

        self._reset_state = None

    def reward(self, cur_obs, action, obs, absorbing):
        dist = self._read_data("goal_pos") - self._read_data("ball_pos")
        return 1. if np.linalg.norm(dist) < 0.05 else 0.

    def is_absorbing(self, state):
        dist = self._read_data("goal_pos") - self._read_data("ball_pos")
        return np.linalg.norm(dist) < 0.05 or self._check_collision("ball", "robot")

    def setup(self, obs):
        if self._reset_state is None:
            # Copy the initial position after the reset
            init_pos = self._data.qpos[0:7].copy()

            # Move the system towards the initial position using a PD-Controller
            for i in range(0, 5000):
                cur_pos = self._data.qpos[0:7].copy()
                cur_vel = self._data.qvel[0:7].copy()

                target = self.linear_movement(init_pos, self.init_robot_pos, 250, i)
                trq = self.p_gains * (target - cur_pos) + self.d_gains * (
                        np.zeros_like(self.init_robot_pos) - cur_vel)
                self._data.ctrl[self._action_indices] = self._data.qfrc_bias[:7] + trq
                mujoco.mj_step(self._model, self._data)

            self._reset_state = deepcopy(self._data)
        else:
            self._data = deepcopy(self._reset_state)

    @staticmethod
    def linear_movement(start, end, n_steps, i):
        t = np.minimum(1., float(i) / float(n_steps))
        return start + (end - start) * t

