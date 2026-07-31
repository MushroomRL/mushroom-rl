import torch
import warp as wp
from pathlib import Path

from mushroom_rl.environments.mujoco_warp import MuJoCoWarp
from mushroom_rl.environments.mujoco import ObservationType
from mushroom_rl.core.spaces import Box


class Walker2DWarp(MuJoCoWarp):
    """
    Mujoco WARP simulation of Walker2d task based on the Hopper environment.
    """

    def __init__(
        self,
        num_envs,
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
        n_substeps=4,
        nconmax=200,
        njmax=200,
        **viewer_params,
    ):
        """
        Constructor.
        """
        xml_path = (
            Path(__file__).resolve().parent.parent
            / "mujoco_envs"
            / "data"
            / "walker_2d"
            / "model.xml"
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
            num_envs=num_envs,
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
        # NOTE: original file said `remove_obs("x_pos", 0)` here but x_pos was
        # never added to observation_spec above — only to additional_data_spec.
        # Matching the hopper pattern (add_obs conditionally) instead.
        if not self._exclude_current_positions_from_observation:
            self.obs_helper.add_obs("x_pos", 1)
        mdp_info = super()._modify_mdp_info(mdp_info)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _create_observation(self, obs):
        obs = obs.clone()

        obs[:, self.obs_helper.joint_vel_idx] = torch.clamp(
            obs[:, self.obs_helper.joint_vel_idx], -10.0, 10.0
        )

        if not self._exclude_current_positions_from_observation:
            x_pos = self._read_data("x_pos")
            obs = torch.cat([obs, x_pos], dim=1)
        return obs

    def _is_within_z_range(self, obs):
        """Check if Z position of torso is within the healthy range."""
        min_z, max_z = self._healthy_z_range
        z_position = self.obs_helper.get_from_obs(obs, "z_pos")[:, 0]
        return (z_position > min_z) & (z_position < max_z)

    def _is_within_angle_range(self, obs):
        """Check if y-angle of torso is within the healthy range."""
        min_angle, max_angle = self._healthy_angle_range
        y_angle = self.obs_helper.get_from_obs(obs, "y_pos")[:, 0]
        return (y_angle > min_angle) & (y_angle < max_angle)

    def _is_healthy(self, obs):
        return self._is_within_z_range(obs) & self._is_within_angle_range(obs)

    def is_absorbing(self, obs):
        return self._terminate_when_unhealthy & ~self._is_healthy(obs)

    def reward(self, obs, action, next_obs, absorbing):
        healthy = self._is_healthy(next_obs)
        healthy_r = (
            healthy | self._terminate_when_unhealthy
        ).float() * self._healthy_reward

        torso_vel = self._read_data("torso_vel")
        forward_r = self._forward_reward_weight * torso_vel[:, 3]

        action_t = torch.as_tensor(
            action, dtype=healthy_r.dtype, device=healthy_r.device
        )
        ctrl_cost = self._ctrl_cost_weight * (action_t**2).sum(dim=-1)

        return healthy_r + forward_r - ctrl_cost

    def setup(self, env_indices, obs):
        """Reset with uniform noise on qpos and qvel for the given environments.

        GPU-native to avoid host-device roundtrips and to keep buffers stable
        for graph capture compatibility.
        """
        super().setup(env_indices, obs)

        qpos = wp.to_torch(self._data_wp.qpos)
        qvel = wp.to_torch(self._data_wp.qvel)

        device = qpos.device
        idx = (
            torch.as_tensor(env_indices, device=device, dtype=torch.long)
            if not isinstance(env_indices, torch.Tensor)
            else env_indices.to(device).long()
        )

        n = len(env_indices)
        noise_pos = (
            torch.rand(n, self._model.nq, device=device) * 2 - 1
        ) * self._reset_noise_scale
        noise_vel = (
            torch.rand(n, self._model.nv, device=device) * 2 - 1
        ) * self._reset_noise_scale

        qpos[idx] += noise_pos
        qvel[idx] += noise_vel

        self._mj_warp.forward(self._model_wp, self._data_wp)

    def _create_info_dictionary(self, obs):
        healthy = self._is_healthy(obs)
        healthy_r = (
            healthy | self._terminate_when_unhealthy
        ).float() * self._healthy_reward
        torso_vel = self._read_data("torso_vel")
        forward_r = self._forward_reward_weight * torso_vel[:, 3]
        return {
            "healthy_reward": healthy_r,
            "forward_reward": forward_r,
        }
