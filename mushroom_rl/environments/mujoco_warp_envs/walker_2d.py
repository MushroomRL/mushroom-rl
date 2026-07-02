import numpy as np
import torch
import warp as wp
import mujoco
from pathlib import Path


from mushroom_rl.environments.mujoco_warp import MuJoCoWarp
from mushroom_rl.environments.mujoco import ObservationType
from mushroom_rl.rl_utils.spaces import Box

class Walker2DWarp(MuJoCo):
    """
    Mujoco WARP simulation of Walker2d task based on the Hopper environment.

    """

    def __init__(
        self,
        gamma=0.99,
        horizon=1000,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.8, 2.0),
        healthy_angle_range=(-1.0, 1.0),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=True,
        n_substeps=n_substeps,
        nconmax=nconmax,
        njmax=njmax,
        **viewer_params,
    ):

        """
        Constructor.

        """

        xml_path = (
            Path(__file__).resolve().parent / "data" / "walker_2d" / "model.xml"
        ).as_posix()
        actuation_spec = [
            "thigh_joint",
            "leg_joint",
            "foot_joint",
            "thigh_left_joint",
            "leg_left_joint",
            "foot_left_joint",
        ]

        observation_spec = [
            ("z_pos", "rootz", ObservationType.JOINT_POS),
            ("y_pos", "rooty", ObservationType.JOINT_POS),
            ("thigh_pos", "thigh_joint", ObservationType.JOINT_POS),
            ("leg_pos", "leg_joint", ObservationType.JOINT_POS),
            ("foot_pos", "foot_joint", ObservationType.JOINT_POS),
            ("thigh_left_pos", "thigh_left_joint", ObservationType.JOINT_POS),
            ("leg_left_pos", "leg_left_joint", ObservationType.JOINT_POS),
            ("foot_left_pos", "foot_left_joint", ObservationType.JOINT_POS),
            ("x_vel", "rootx", ObservationType.JOINT_VEL),
            ("z_vel", "rootz", ObservationType.JOINT_VEL),
            ("y_vel", "rooty", ObservationType.JOINT_VEL),
            ("thigh_vel", "thigh_joint", ObservationType.JOINT_VEL),
            ("leg_vel", "leg_joint", ObservationType.JOINT_VEL),
            ("foot_vel", "foot_joint", ObservationType.JOINT_VEL),
            ("thigh_left_vel", "thigh_left_joint", ObservationType.JOINT_VEL),
            ("leg_left_vel", "leg_left_joint", ObservationType.JOINT_VEL),
            ("foot_left_vel", "foot_left_joint", ObservationType.JOINT_VEL),
        ]

        additional_data_spec = [
            ("x_pos", "rootx", ObservationType.JOINT_POS),
            ("torso_vel", "torso", ObservationType.BODY_VEL_WORLD),
        ]

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        
        super().__init__(
            xml_file=xml_path,
            gamma=gamma,
            horizon=horizon,
            observation_spec=observation_spec,
            actuation_spec=actuation_spec,
            additional_data_spec=additional_data_spec,
            n_substeps=n_substeps,
            nconmax=nconmax,
            njmax=njmax,
            **viewer_params,
        )

    def _modify_mdp_info(self, mdp_info):
        if not self._exclude_current_positions_from_observation:
            self.obs_helper.remove_obs("x_pos", 0)
        mdp_info = super()._modify_mdp_info(mdp_info)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _create_observation(self, obs):
        obs = obs.clone()

        obs[:, self.obs_helper.joint_vel_idx] = torch.clamp(
            obs[:, self.obs_helper.joint_vel_idx], -10.0, 10.0
        )

        if not self._exclude_current_positions_from_observations:
            x_pos = self._read_data("x_pos")
            obs = torch.cat([obs, x_pos], dim = 1)
        return obs

    def _is_within_z_range(self, obs):

        pass

    def _is_within_angle_range(self,obs):
        pass

    def is_absorbing(self, obs):
        pass

    def _is_health(self, obs):
        pass

    def _get_healthy_reward(self, obs):
        pass

    def _get_forward_reward(self):
        pass

    def _get_ctrl_cost(self, action):
        pass

    def reward(self, obs, action, next_obs, absorbing):
        pass

    def _generate_noise(self):
        pass

    def setup(self, obs):
        pass

    def _create_info_dictionary(self, obs, action):
        pass




