from pathlib import Path

import numpy as np
import mujoco

from mushroom_rl.environments.mujoco import ObservationType
from mushroom_rl.rl_utils.spaces import Box
from mushroom_rl.utils.quaternions import quaternion_distance
from mushroom_rl.utils.angles import euler_to_quat
from mushroom_rl.environments.mujoco_envs.panda import Panda


class Reach(Panda):

    def __init__(
        self,
        gamma: float = 0.99,
        horizon: int = 200,
        gripper_goal_distance_reward_weight: float = -2.0,
        gripper_goal_rotation_reward_weight: float = -1.0,
        ctrl_cost_weight: float = -1e-4,
        n_substeps: int = 5,
        **viewer_params,
    ):

        xml_path = (
            Path(__file__).resolve().parent / "data" / "panda" / "reach.xml"
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
            ("goal_pos", "goal", ObservationType.BODY_POS),
            ("goal_rot", "goal", ObservationType.BODY_ROT),
        ]

        self._gripper_goal_distance_reward_weight = gripper_goal_distance_reward_weight
        self._gripper_goal_rotation_reward_weight = gripper_goal_rotation_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        viewer_params = viewer_params or {}
        viewer_params.setdefault("reference_frame_visualization_on_startup", 3)

        super().__init__(
            xml_path,
            gamma=gamma,
            horizon=horizon,
            actuation_spec=actuation_spec,
            additional_data_spec=additional_data_spec,
            n_substeps=n_substeps,
            **viewer_params,
        )

    def _modify_mdp_info(self, mdp_info):
        mdp_info = super()._modify_mdp_info(mdp_info)
        self.obs_helper.add_obs("rel_goal_pos", 3)
        self.obs_helper.add_obs("goal_rot", 4)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _create_observation(self, obs):
        obs = super()._create_observation(obs)

        gripper_pos = self._read_data("gripper_pos")
        goal_pos = self._read_data("goal_pos")
        rel_goal_pos = goal_pos - gripper_pos
        goal_rot = self._read_data("goal_rot")

        obs = np.concatenate([obs, rel_goal_pos, goal_rot])
        return obs

    def _get_gripper_goal_distance_reward(self, obs):
        rel_goal_pos = self.obs_helper.get_from_obs(obs, "rel_goal_pos")
        gripper_goal_distance = np.linalg.norm(rel_goal_pos)
        return self._gripper_goal_distance_reward_weight * gripper_goal_distance

    def _get_gripper_goal_rotation_reward(self, obs):
        gripper_rot = self.obs_helper.get_from_obs(obs, "gripper_rot")
        goal_rot = self.obs_helper.get_from_obs(obs, "goal_rot")
        gripper_goal_rotation = quaternion_distance(gripper_rot, goal_rot)
        return self._gripper_goal_rotation_reward_weight * gripper_goal_rotation

    def _get_ctrl_cost(self, action):
        ctrl_cost = np.sum(np.square(action))
        return self._ctrl_cost_weight * ctrl_cost

    def reward(self, obs, action, next_obs, absorbing):
        gripper_goal_distance_reward = self._get_gripper_goal_distance_reward(next_obs)
        gripper_goal_rotation_reward = self._get_gripper_goal_rotation_reward(next_obs)
        ctrl_cost = self._get_ctrl_cost(action)
        reward = gripper_goal_distance_reward + gripper_goal_rotation_reward + ctrl_cost
        return reward

    def is_absorbing(self, obs):
        return False

    def _randomize_goal_pose(self):
        pose_range = {
            "x": (0.35, 0.65),
            "y": (-0.2, 0.2),
            "z": (0.15, 0.5),
            "roll": (0.0, 0.0),
            "pitch": (3.14, 3.14),
            "yaw": (-3.14, 3.14),
        }
        mocap_id = self._model.body("goal").mocapid[0]
        self._data.mocap_pos[mocap_id][0] = np.random.uniform(*pose_range["x"])
        self._data.mocap_pos[mocap_id][1] = np.random.uniform(*pose_range["y"])
        self._data.mocap_pos[mocap_id][2] = np.random.uniform(*pose_range["z"])

        roll = np.random.uniform(*pose_range["roll"])
        pitch = np.random.uniform(*pose_range["pitch"])
        yaw = np.random.uniform(*pose_range["yaw"])
        self._data.mocap_quat[0][:] = euler_to_quat(np.array([roll, pitch, yaw]))

    def setup(self, obs):
        super().setup(obs)
        self._randomize_goal_pose()
        mujoco.mj_forward(self._model, self._data)  # type: ignore

    def _create_info_dictionary(self, obs, action):
        info = super()._create_info_dictionary(obs, action)
        info["gripper_goal_distance"] = np.linalg.norm(
            self.obs_helper.get_from_obs(obs, "rel_goal_pos")
        )
        info["gripper_goal_rotation"] = quaternion_distance(
            self.obs_helper.get_from_obs(obs, "gripper_rot"),
            self.obs_helper.get_from_obs(obs, "goal_rot"),
        )
        info["gripper_goal_distance_reward"] = self._get_gripper_goal_distance_reward(
            obs
        )
        info["gripper_goal_rotation_reward"] = self._get_gripper_goal_rotation_reward(
            obs
        )
        info["ctrl_cost"] = self._get_ctrl_cost(action)
        return info
