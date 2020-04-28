from mushroom_rl.environments.mujoco import MuJoCo, ObservationType
import numpy as np
from pathlib import Path


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

        observation_spec = [("wam/base_yaw_joint", ObservationType.JOINT_POS),
                            ("wam/base_yaw_joint", ObservationType.JOINT_VEL),
                            ("wam/shoulder_pitch_joint", ObservationType.JOINT_POS),
                            ("wam/shoulder_pitch_joint", ObservationType.JOINT_VEL),
                            ("wam/shoulder_yaw_joint", ObservationType.JOINT_POS),
                            ("wam/shoulder_yaw_joint", ObservationType.JOINT_VEL),
                            ("wam/elbow_pitch_joint", ObservationType.JOINT_POS),
                            ("wam/elbow_pitch_joint", ObservationType.JOINT_VEL),
                            ("wam/wrist_yaw_joint", ObservationType.JOINT_POS),
                            ("wam/wrist_yaw_joint", ObservationType.JOINT_VEL),
                            ("wam/wrist_pitch_joint", ObservationType.JOINT_POS),
                            ("wam/wrist_pitch_joint", ObservationType.JOINT_VEL),
                            ("wam/palm_yaw_joint", ObservationType.JOINT_POS),
                            ("wam/palm_yaw_joint", ObservationType.JOINT_VEL),
                            ("ball", ObservationType.BODY_POS),
                            ("ball", ObservationType.BODY_VEL)]

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

        super().__init__(xml_path, action_spec, observation_spec, 0.9999, 2000, n_substeps=4,
                         additional_data_spec=additional_data_spec, collision_groups=collision_groups)

        self.init_robot_pos = np.array([0.0, 0.58760536, 0.0, 1.36004913, 0.0, -0.32072943, -1.57])
        self.p_gains = np.array([200, 300, 100, 100, 10, 10, 2.5])
        self.d_gains = np.array([7, 15, 5, 2.5, 0.3, 0.3, 0.05])

    def _reward(self, state, action, next_state):
        dist = self._read_data("goal_pos") - self._read_data("ball_pos")
        return 1. if np.linalg.norm(dist) < 0.05 else 0.

    def _is_absorbing(self, state):
        dist = self._read_data("goal_pos") - self._read_data("ball_pos")
        return np.linalg.norm(dist) < 0.05 or self._check_collision("ball", "robot")

    def _setup(self):
        # Copy the initial position after the reset
        init_pos = self._sim.data.qpos.copy()
        init_vel = np.zeros_like(init_pos)

        # Reset the system and the set the intial robot position
        self._sim.data.qpos[:] = init_pos
        self._sim.data.qvel[:] = init_vel
        self._sim.data.qpos[0:7] = self.init_robot_pos

        # Do one simulation step to compute the new position of the goal_site
        self._sim.step()

        self._sim.data.qpos[:] = init_pos
        self._sim.data.qvel[:] = init_vel
        self._sim.data.qpos[0:7] = self.init_robot_pos
        self._write_data("ball_pos", self._read_data("goal_pos") - np.array([0., 0., 0.329]))

        # Stabilize the system around the initial position using a PD-Controller
        for i in range(0, 500):
            self._sim.data.qpos[7:] = 0.
            self._sim.data.qvel[7:] = 0.
            self._sim.data.qpos[7] = -0.2
            cur_pos = self._sim.data.qpos[0:7].copy()
            cur_vel = self._sim.data.qvel[0:7].copy()
            trq = self.p_gains * (self.init_robot_pos - cur_pos) + self.d_gains * (
                    np.zeros_like(self.init_robot_pos) - cur_vel)
            self._sim.data.qfrc_applied[0:7] = trq
            self._sim.step()

        # Now simulate for more time-steps without resetting the position of the first link of the rope
        for i in range(0, 500):
            cur_pos = self._sim.data.qpos[0:7].copy()
            cur_vel = self._sim.data.qvel[0:7].copy()
            trq = self.p_gains * (self.init_robot_pos - cur_pos) + self.d_gains * (
                    np.zeros_like(self.init_robot_pos) - cur_vel)
            self._sim.data.qfrc_applied[0:7] = trq
            self._sim.step()
