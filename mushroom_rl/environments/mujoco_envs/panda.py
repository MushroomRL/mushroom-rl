import math
import random
from pathlib import Path

import mujoco
import numpy as np

from mushroom_rl.environments.mujoco import MuJoCo, ObservationType, MujocoViewer
from mushroom_rl.rl_utils.spaces import Box
from mushroom_rl.utils.quaternions import mat_to_quat


class Panda(MuJoCo):

    def __init__(
        self,
        xml_path,
        gamma,
        horizon,
        n_substeps,
        additional_data_spec=None,
        collision_groups=None,
        actuation_spec=None,
        keyframe="home",
        **viewer_params,
    ):
        actuation_spec = actuation_spec or [
            "actuator1",
            "actuator2",
            "actuator3",
            "actuator4",
            "actuator5",
            "actuator6",
            "actuator7",
            "actuator8",
        ]

        observation_spec = [
            ("joint1_pos", "joint1", ObservationType.JOINT_POS),
            ("joint2_pos", "joint2", ObservationType.JOINT_POS),
            ("joint3_pos", "joint3", ObservationType.JOINT_POS),
            ("joint4_pos", "joint4", ObservationType.JOINT_POS),
            ("joint5_pos", "joint5", ObservationType.JOINT_POS),
            ("joint6_pos", "joint6", ObservationType.JOINT_POS),
            ("joint7_pos", "joint7", ObservationType.JOINT_POS),
            ("finger_joint1_pos", "finger_joint1", ObservationType.JOINT_POS),
            ("finger_joint2_pos", "finger_joint2", ObservationType.JOINT_POS),
            ("gripper_pos", "gripper", ObservationType.SITE_POS),
            ("joint1_vel", "joint1", ObservationType.JOINT_VEL),
            ("joint2_vel", "joint2", ObservationType.JOINT_VEL),
            ("joint3_vel", "joint3", ObservationType.JOINT_VEL),
            ("joint4_vel", "joint4", ObservationType.JOINT_VEL),
            ("joint5_vel", "joint5", ObservationType.JOINT_VEL),
            ("joint6_vel", "joint6", ObservationType.JOINT_VEL),
            ("joint7_vel", "joint7", ObservationType.JOINT_VEL),
            ("finger_joint1_vel", "finger_joint1", ObservationType.JOINT_VEL),
            ("finger_joint2_vel", "finger_joint2", ObservationType.JOINT_VEL),
        ]

        additional_data_spec = additional_data_spec or []

        additional_data_spec += [
            ("joint1_pos", "joint1", ObservationType.JOINT_POS),
            ("joint2_pos", "joint2", ObservationType.JOINT_POS),
            ("joint3_pos", "joint3", ObservationType.JOINT_POS),
            ("joint4_pos", "joint4", ObservationType.JOINT_POS),
            ("joint5_pos", "joint5", ObservationType.JOINT_POS),
            ("joint6_pos", "joint6", ObservationType.JOINT_POS),
            ("joint7_pos", "joint7", ObservationType.JOINT_POS),
            ("finger_joint1_pos", "finger_joint1", ObservationType.JOINT_POS),
            ("finger_joint2_pos", "finger_joint2", ObservationType.JOINT_POS),
            ("gripper_pos", "gripper", ObservationType.SITE_POS),
            ("gripper_rot", "gripper", ObservationType.SITE_ROT),
        ]

        collision_groups = collision_groups or []

        collision_groups += [
            ("hand", ["hand_c"]),
            (
                "gripper",
                [
                    "hand_c",
                    "left_fingertip_pad_collision_1",
                    "left_fingertip_pad_collision_2",
                    "left_fingertip_pad_collision_3",
                    "left_fingertip_pad_collision_4",
                    "left_fingertip_pad_collision_5",
                    "right_fingertip_pad_collision_1",
                    "right_fingertip_pad_collision_2",
                    "right_fingertip_pad_collision_3",
                    "right_fingertip_pad_collision_4",
                    "right_fingertip_pad_collision_5",
                ],
            ),
            (
                "robot",
                [
                    "link0_c",
                    "link1_c",
                    "link2_c",
                    "link3_c",
                    "link4_c",
                    "link5_c0",
                    "link5_c1",
                    "link5_c2",
                    "link6_c",
                    "link7_c",
                ],
            ),
            ("floor", ["floor"]),
        ]

        self._keyframe = keyframe

        viewer_params = viewer_params or {}
        viewer_params.setdefault(
            "camera_params", MujocoViewer.get_default_camera_params()
        )

        viewer_params["camera_params"]["static"].update(
            {
                "distance": 3,
                "elevation": -20.0,
                "azimuth": 90.0,
                "lookat": np.array([0.5, 0.0, 0.0]),
            }
        )

        super().__init__(
            xml_path,
            gamma=gamma,
            horizon=horizon,
            actuation_spec=actuation_spec,
            observation_spec=observation_spec,
            additional_data_spec=additional_data_spec,
            collision_groups=collision_groups,
            n_substeps=n_substeps,
            **viewer_params,
        )

    def _modify_mdp_info(self, mdp_info):
        self.obs_helper.add_obs("gripper_rot", 4)

        mdp_info = super()._modify_mdp_info(mdp_info)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _create_observation(self, obs):
        obs = super()._create_observation(obs)

        gripper_rot = self._read_data("gripper_rot").reshape(3, 3)
        gripper_rot = mat_to_quat(gripper_rot)

        obs = np.concatenate([obs, gripper_rot])
        return obs

    def _get_contact_force(self, group1, group2, contact_force_range):
        collision_force = self._get_collision_force(group1, group2)
        contact_force = np.clip(collision_force, *contact_force_range)
        contact_force = np.sum(np.square(contact_force), keepdims=True)
        return contact_force

    def _load_keyframe(self, name: str):
        keyframe = self._model.keyframe(name)
        mujoco.mj_resetDataKeyframe(self._model, self._data, keyframe.id)  # type: ignore

    def setup(self, obs):
        super().setup(obs)
        self._load_keyframe(self._keyframe)

    # Gravity compensation implementation adapted from https://colab.research.google.com/drive/1zlsplgSyk59hxnw3kOJMIxAXuwxXqOHD?usp=sharing
    def get_body_children_ids(self, body_id):
        return [
            i
            for i in range(self._model.nbody)
            if self._model.body_parentid[i] == body_id
            and body_id != i  # Exclude the body itself.
        ]

    def get_subtree_body_ids(self, body_id):
        body_ids: list[int] = []
        stack = [body_id]
        while stack:
            body_id = stack.pop()
            body_ids.append(body_id)
            stack += self.get_body_children_ids(body_id)
        return body_ids

    def gravity_compensation(
        self,
        subtree_body_id,
    ) -> None:
        self._data.qfrc_applied[:] = 0.0
        jac = np.empty((3, self._model.nv))
        total_mass = self._model.body_subtreemass[subtree_body_id]
        mujoco.mj_jacSubtreeCom(self._model, self._data, jac, subtree_body_id)
        self._data.qfrc_applied[:] -= self._model.opt.gravity * total_mass @ jac

    def _simulation_pre_step(self):
        super()._simulation_pre_step()
        subtree_body_id = self._model.body("link0").id
        self.gravity_compensation(subtree_body_id)
