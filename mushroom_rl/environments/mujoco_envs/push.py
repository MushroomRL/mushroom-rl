from pathlib import Path

import numpy as np
import mujoco

from mushroom_rl.environments.mujoco import ObservationType
from mushroom_rl.rl_utils.spaces import Box
from mushroom_rl.environments.mujoco_envs.panda import Panda


class Push(Panda):
    def __init__(
        self,
        gamma: float = 0.99,
        horizon: int = 200,
        gripper_cube_distance_reward_weight: float = -1.0,
        cube_goal_distance_reward_weight: float = -2.0,
        ctrl_cost_weight: float = -1e-4,
        contact_cost_weight: float = -1e-4,
        n_substeps: int = 5,
        contact_force_range: tuple[float, float] = (-1.0, 1.0),
        **viewer_params,
    ):

        xml_path = (
            Path(__file__).resolve().parent / "data" / "panda" / "push.xml"
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
            ("cube_pose", "cube", ObservationType.JOINT_POS),
            ("goal_pos", "goal", ObservationType.BODY_POS),
        ]

        collision_groups = [
            ("cube", ["cube"]),
            ("table", ["table"]),
        ]

        self._gripper_cube_distance_reward_weight = gripper_cube_distance_reward_weight
        self._cube_goal_distance_reward_weight = cube_goal_distance_reward_weight
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

        self.obs_helper.add_obs("rel_cube_pos", 3)
        self.obs_helper.add_obs("rel_goal_pos", 3)
        self.obs_helper.add_obs("contact_force", 1)

        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _create_observation(self, obs):
        obs = super()._create_observation(obs)

        gripper_pos = self._read_data("gripper_pos")
        cube_pose = self._read_data("cube_pose")
        cube_pos = cube_pose[:3]
        goal_pos = self._read_data("goal_pos")

        rel_cube_pos = cube_pos - gripper_pos
        rel_goal_pos = goal_pos - cube_pos
        contact_force = self._get_contact_force(
            "robot", "table", self._contact_force_range
        ) + self._get_contact_force("gripper", "table", self._contact_force_range)

        obs = np.concatenate([obs, rel_cube_pos, rel_goal_pos, contact_force])
        return obs

    def _get_gripper_cube_distance_reward(self, obs):
        rel_cube_pos = self.obs_helper.get_from_obs(obs, "rel_cube_pos")
        gripper_cube_distance = np.linalg.norm(rel_cube_pos)
        return self._gripper_cube_distance_reward_weight * gripper_cube_distance

    def _get_cube_goal_distance_reward(self, obs):
        rel_goal_pos = self.obs_helper.get_from_obs(obs, "rel_goal_pos")
        cube_goal_distance = np.linalg.norm(rel_goal_pos)
        return self._cube_goal_distance_reward_weight * cube_goal_distance

    def _get_ctrl_cost(self, action):
        ctrl_cost = np.sum(np.square(action))
        return self._ctrl_cost_weight * ctrl_cost

    def _get_contact_cost(self, obs):
        contact_force = self.obs_helper.get_from_obs(obs, "contact_force")
        return self._contact_cost_weight * contact_force

    def reward(self, obs, action, next_obs, absorbing):
        gripper_cube_distance_reward = self._get_gripper_cube_distance_reward(obs)
        cube_goal_distance_reward = self._get_cube_goal_distance_reward(obs)
        ctrl_cost = self._get_ctrl_cost(action)
        contact_cost = self._get_contact_cost(next_obs)
        reward = (
            cube_goal_distance_reward
            + gripper_cube_distance_reward
            + ctrl_cost
            + contact_cost
        )
        return reward

    def is_absorbing(self, obs):
        return False

    def _randomize_cube_pos(self):
        pos_range = {"x": (0.4, 0.6), "y": (0.1, 0.25)}

        cube_pose = self._read_data("cube_pose")
        cube_pose[0] = np.random.uniform(*pos_range["x"])
        cube_pose[1] = np.random.uniform(*pos_range["y"])
        self._write_data("cube_pose", cube_pose)

    def _randomize_goal_pos(self):
        pos_range = {"x": (0.4, 0.6), "y": (-0.1, -0.25)}
        mocap_id = self._model.body("goal").mocapid[0]
        self._data.mocap_pos[mocap_id][0] = np.random.uniform(*pos_range["x"])
        self._data.mocap_pos[mocap_id][1] = np.random.uniform(*pos_range["y"])

    def setup(self, obs):
        super().setup(obs)
        self._randomize_cube_pos()
        self._randomize_goal_pos()
        mujoco.mj_forward(self._model, self._data)  # type: ignore

    def _create_info_dictionary(self, obs, action):
        info = super()._create_info_dictionary(obs, action)
        info["gripper_cube_distance"] = np.linalg.norm(
            self.obs_helper.get_from_obs(obs, "rel_cube_pos")
        )
        info["cube_goal_distance"] = np.linalg.norm(
            self.obs_helper.get_from_obs(obs, "rel_goal_pos")
        )
        info["gripper_cube_distance_reward"] = self._get_gripper_cube_distance_reward(
            obs
        )
        info["cube_goal_distance_reward"] = self._get_cube_goal_distance_reward(obs)
        info["ctrl_cost"] = self._get_ctrl_cost(action)
        info["contact_cost"] = self._get_contact_cost(obs)
        return info
