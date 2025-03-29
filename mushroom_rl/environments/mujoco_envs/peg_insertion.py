from pathlib import Path

import numpy as np
import mujoco

from mushroom_rl.environments.mujoco import ObservationType
from mushroom_rl.rl_utils.spaces import Box
from mushroom_rl.utils.quaternions import quaternion_distance
from mushroom_rl.environments.mujoco_envs.panda import Panda


class PegInsertion(Panda):
    def __init__(
        self,
        gamma: float = 0.99,
        horizon: int = 300,
        alignment_reward_weight: float = 1.0,
        insertion_reward_weight: float = 15.0,
        rotation_reward_weight: float = 2.0,
        ctrl_cost_weight: float = -1e-4,
        contact_cost_weight: float = 0,
        n_substeps: int = 5,
        contact_force_range: tuple[float, float] = (-1.0, 1.0),
        **viewer_params,
    ):
        xml_path = (
            Path(__file__).resolve().parent / "data" / "panda" / "peg_insertion.xml"
        ).as_posix()

        actuation_spec = [
            "actuator1",
            "actuator2",
            "actuator3",
            "actuator4",
            "actuator5",
            "actuator6",
            "actuator7",
        ]

        additional_data_spec = [
            ("peg_pos", "peg", ObservationType.SITE_POS),
            ("peg_rot", "peg", ObservationType.BODY_ROT),
            ("goal_pos", "hole", ObservationType.SITE_POS),
            ("goal_rot", "hole", ObservationType.BODY_ROT),
            ("goal_pose", "hole", ObservationType.JOINT_POS),
        ]

        collision_groups = [
            ("peg", ["peg"]),
            ("table", ["table"]),
        ]

        self._alignment_reward_weight = alignment_reward_weight
        self._insertion_reward_weight = insertion_reward_weight
        self._rotation_reward_weight = rotation_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._contact_force_range = contact_force_range

        super().__init__(
            xml_path,
            gamma=gamma,
            horizon=horizon,
            actuation_spec=actuation_spec,
            additional_data_spec=additional_data_spec,
            collision_groups=collision_groups,
            n_substeps=n_substeps,
            **viewer_params,
        )

    def _modify_mdp_info(self, mdp_info):
        mdp_info = super()._modify_mdp_info(mdp_info)

        self.obs_helper.add_obs("rel_peg_pos", 3)
        self.obs_helper.add_obs("peg_rot", 4)
        self.obs_helper.add_obs("rel_goal_pos", 3)
        self.obs_helper.add_obs("goal_rot", 4)
        self.obs_helper.add_obs("collision_force", 1)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _create_observation(self, obs):
        obs = super()._create_observation(obs)

        gripper_pos = self._read_data("gripper_pos")
        peg_pos = self._read_data("peg_pos")
        peg_rot = self._read_data("peg_rot")
        goal_pos = self._read_data("goal_pos")
        goal_rot = self._read_data("goal_rot")

        rel_peg_pos = peg_pos - gripper_pos
        rel_goal_pos = goal_pos - peg_pos
        collision_force = self._get_contact_force(
            "robot", "table", self._contact_force_range
        ) + self._get_contact_force("gripper", "table", self._contact_force_range)

        obs = np.concatenate(
            [
                obs,
                rel_peg_pos,
                peg_rot,
                rel_goal_pos,
                goal_rot,
                collision_force,
            ]
        )
        return obs

    def _is_aligned(self, obs):
        rel_xy_goal_pos = self.obs_helper.get_from_obs(self._obs, "rel_goal_pos")[:2]
        alignment = np.linalg.norm(rel_xy_goal_pos)
        return alignment < 0.0025

    def _is_rotated(self, obs):
        peg_rotation = self.obs_helper.get_from_obs(obs, "peg_rot")
        goal_rotation = self.obs_helper.get_from_obs(obs, "goal_rot")
        peg_goal_rotation = quaternion_distance(peg_rotation, goal_rotation)
        return peg_goal_rotation < 0.01

    def _get_alignment_reward(self, obs):
        rel_xy_goal_pos = self.obs_helper.get_from_obs(obs, "rel_goal_pos")[:2]
        peg_goal_alignment = np.linalg.norm(rel_xy_goal_pos).item()
        return self._alignment_reward_weight * (1 - np.tanh(peg_goal_alignment / 0.1))

    def _get_insertion_reward(self, obs):
        rel_z_goal_pos = self.obs_helper.get_from_obs(obs, "rel_goal_pos")[2]
        peg_goal_insertion = np.linalg.norm(rel_z_goal_pos).item()
        return (
            self._insertion_reward_weight
            * self._is_aligned(obs)
            * (1 - np.tanh(peg_goal_insertion / 0.1))
        )

    def _get_rotation_reward(self, obs):
        peg_rotation = self.obs_helper.get_from_obs(obs, "peg_rot")
        goal_rotation = self.obs_helper.get_from_obs(obs, "goal_rot")
        peg_goal_rotation = quaternion_distance(peg_rotation, goal_rotation)
        return self._rotation_reward_weight * (1 - np.tanh(peg_goal_rotation / 0.1))

    def _get_ctrl_cost(self, action):
        ctrl_cost = np.sum(np.square(action))
        return self._ctrl_cost_weight * ctrl_cost

    def _get_contact_cost(self, obs):
        collision_force = self.obs_helper.get_from_obs(obs, "collision_force")
        return self._contact_cost_weight * collision_force

    def reward(self, obs, action, next_obs, absorbing):
        alignment_reward = self._get_alignment_reward(next_obs)
        insertion_reward = self._get_insertion_reward(next_obs)
        rotation_reward = self._get_rotation_reward(next_obs)
        ctrl_cost = self._get_ctrl_cost(action)
        contact_cost = self._get_contact_cost(next_obs)

        reward = (
            alignment_reward
            + insertion_reward
            + rotation_reward
            + ctrl_cost
            + contact_cost
        )

        return reward

    def is_absorbing(self, obs):
        return not self._check_collision("gripper", "peg")

    def _randomize_goal_pos(self):
        pose_range = {"x": (0.4, 0.6), "y": (-0.25, 0.25)}
        mocap_id = self._model.body("hole").mocapid[0]
        self._data.mocap_pos[mocap_id][0] = np.random.uniform(*pose_range["x"])
        self._data.mocap_pos[mocap_id][1] = np.random.uniform(*pose_range["y"])

    def setup(self, obs):
        super().setup(obs)
        self._randomize_goal_pos()
        mujoco.mj_forward(self._model, self._data)  # type: ignore

    def _create_info_dictionary(self, obs, action):
        info = super()._create_info_dictionary(obs, action)
        rel_goal_pos = self.obs_helper.get_from_obs(obs, "rel_goal_pos")
        info["alignment"] = np.linalg.norm(rel_goal_pos[:2]).item()
        info["insertion"] = np.linalg.norm(rel_goal_pos[2]).item()
        info["rotation"] = quaternion_distance(
            self.obs_helper.get_from_obs(obs, "peg_rot"),
            self.obs_helper.get_from_obs(obs, "goal_rot"),
        )
        info["alignment_reward"] = self._get_alignment_reward(obs)
        info["insertion_reward"] = self._get_insertion_reward(obs)
        info["rotation_reward"] = self._get_rotation_reward(obs)
        info["ctrl_cost"] = self._get_ctrl_cost(action)
        info["contact_cost"] = self._get_contact_cost(obs)
        return info
