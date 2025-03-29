from pathlib import Path

import numpy as np
import mujoco

from mushroom_rl.environments.mujoco import ObservationType
from mushroom_rl.rl_utils.spaces import Box
from mushroom_rl.utils.quaternions import quaternion_distance
from mushroom_rl.environments.mujoco_envs.panda import Panda


class Pick(Panda):
    def __init__(
        self,
        gamma: float = 0.99,
        horizon: int = 200,
        gripper_cube_distance_reward_weight: float = 1.0,
        cube_goal_distance_reward_weight: float = 20.0,
        cube_goal_rotation_reward_weight: float = 10.0,
        ctrl_cost_weight: float = -1e-4,
        contact_cost_weight: float = -1e-4,
        n_substeps: int = 5,
        contact_force_range: tuple[float, float] = (-1.0, 1.0),
        **viewer_params,
    ):
        xml_path = (
            Path(__file__).resolve().parent / "data" / "panda" / "pick.xml"
        ).as_posix()

        additional_data_spec = [
            ("cube_pose", "cube", ObservationType.JOINT_POS),
            ("goal_pos", "goal", ObservationType.BODY_POS),
            ("goal_rot", "goal", ObservationType.BODY_ROT),
        ]

        collision_groups = [
            ("cube", ["cube"]),
            ("table", ["table"]),
        ]

        self._gripper_cube_distance_reward_weight = gripper_cube_distance_reward_weight
        self._cube_goal_distance_reward_weight = cube_goal_distance_reward_weight
        self._cube_goal_rotation_reward_weight = cube_goal_rotation_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._contact_force_range = contact_force_range

        super().__init__(
            xml_path,
            gamma=gamma,
            horizon=horizon,
            additional_data_spec=additional_data_spec,
            collision_groups=collision_groups,
            n_substeps=n_substeps,
            **viewer_params,
        )

    def _modify_mdp_info(self, mdp_info):
        mdp_info = super()._modify_mdp_info(mdp_info)

        self.obs_helper.add_obs("rel_cube_pos", 3)
        self.obs_helper.add_obs("cube_rot", 4)
        self.obs_helper.add_obs("rel_goal_pos", 3)
        self.obs_helper.add_obs("goal_rot", 4)
        self.obs_helper.add_obs("contact_force", 1)

        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _create_observation(self, obs):
        obs = super()._create_observation(obs)

        gripper_pos = self._read_data("gripper_pos")
        cube_pose = self._read_data("cube_pose")
        cube_pos, cube_rot = cube_pose[:3], cube_pose[3:]
        goal_pos = self._read_data("goal_pos")
        goal_rot = self._read_data("goal_rot")

        rel_cube_pos = cube_pos - gripper_pos
        rel_goal_pos = goal_pos - cube_pos

        contact_force = self._get_contact_force(
            "robot", "table", self._contact_force_range
        ) + self._get_contact_force("gripper", "table", self._contact_force_range)

        obs = np.concatenate(
            [
                obs,
                rel_cube_pos,
                cube_rot,
                rel_goal_pos,
                goal_rot,
                contact_force,
            ]
        )

        return obs

    def _is_cube_lifted(self):
        cube_pose = self._read_data("cube_pose")
        return cube_pose[2] > 0.04

    def _get_gripper_cube_distance_reward(self, obs):
        rel_cube_pos = self.obs_helper.get_from_obs(obs, "rel_cube_pos")
        gripper_cube_distance = np.linalg.norm(rel_cube_pos)
        return self._gripper_cube_distance_reward_weight * (
            1 - np.tanh(gripper_cube_distance / 0.1)
        )

    def _get_cube_goal_distance_reward(self, obs):
        rel_goal_pos = self.obs_helper.get_from_obs(obs, "rel_goal_pos")
        cube_goal_distance = np.linalg.norm(rel_goal_pos)
        return (
            self._cube_goal_distance_reward_weight
            * self._is_cube_lifted()
            * (1 - np.tanh(cube_goal_distance / 0.4))
        )

    def _get_cube_goal_rotation_reward(self, obs):
        cube_rot = self.obs_helper.get_from_obs(obs, "cube_rot")
        goal_rot = self.obs_helper.get_from_obs(obs, "goal_rot")
        cube_goal_rotation = quaternion_distance(cube_rot, goal_rot)
        return (
            self._cube_goal_rotation_reward_weight
            * self._is_cube_lifted()
            * (1 - np.tanh(cube_goal_rotation / 0.3))
        )

    def _get_ctrl_cost(self, action):
        ctrl_cost = np.sum(np.square(action))
        return self._ctrl_cost_weight * ctrl_cost

    def _get_contact_cost(self, obs):
        contact_force = self.obs_helper.get_from_obs(obs, "contact_force")
        return self._contact_cost_weight * contact_force

    def reward(self, obs, action, next_obs, absorbing):
        gripper_cube_distance_reward = self._get_gripper_cube_distance_reward(next_obs)
        cube_goal_distance_reward = self._get_cube_goal_distance_reward(next_obs)
        cube_goal_rotation_reward = self._get_cube_goal_rotation_reward(next_obs)
        ctrl_cost = self._get_ctrl_cost(action)
        contact_cost = self._get_contact_cost(next_obs)

        reward = (
            gripper_cube_distance_reward
            + cube_goal_distance_reward
            + cube_goal_rotation_reward
            + ctrl_cost
            + contact_cost
        )
        return reward

    def is_absorbing(self, obs):
        return self._check_collision("cube", "floor")

    def _randomize_cube_pos(self):
        pose_range = {"x": (0.4, 0.6), "y": (-0.25, 0.25)}
        cube_pose = self._read_data("cube_pose")
        cube_pose[0] = np.random.uniform(*pose_range["x"])
        cube_pose[1] = np.random.uniform(*pose_range["y"])
        self._write_data("cube_pose", cube_pose)

    def _randomize_goal_pos(self):
        pose_range = {"x": (0.4, 0.6), "y": (-0.25, 0.25), "z": (0.25, 0.5)}
        mocap_id = self._model.body("goal").mocapid[0]
        self._data.mocap_pos[mocap_id][0] = np.random.uniform(*pose_range["x"])
        self._data.mocap_pos[mocap_id][1] = np.random.uniform(*pose_range["y"])
        self._data.mocap_pos[mocap_id][2] = np.random.uniform(*pose_range["z"])

    def setup(self, obs):
        super().setup(obs)
        self._randomize_cube_pos()
        self._randomize_goal_pos()
        mujoco.mj_forward(self._model, self._data)  # type: ignore

    def _create_info_dictionary(self, obs, action):
        info = super()._create_info_dictionary(obs, action)
        info["gripper_cube_distance_reward"] = self._get_gripper_cube_distance_reward(
            obs
        )
        info["cube_goal_distance_reward"] = self._get_cube_goal_distance_reward(obs)
        info["cube_goal_rotation_reward"] = self._get_cube_goal_rotation_reward(obs)
        info["cube_goal_distance"] = np.linalg.norm(
            self.obs_helper.get_from_obs(obs, "rel_goal_pos")
        )
        info["cube_goal_rotation"] = quaternion_distance(
            self.obs_helper.get_from_obs(obs, "cube_rot"),
            self.obs_helper.get_from_obs(obs, "goal_rot"),
        )
        info["gripper_cube_distance"] = np.linalg.norm(
            self.obs_helper.get_from_obs(obs, "rel_cube_pos")
        )
        info["cube_z_pos"] = self._read_data("cube_pose")[2]
        info["ctrl_cost"] = self._get_ctrl_cost(action)
        info["contact_cost"] = self._get_contact_cost(obs)
        return info
