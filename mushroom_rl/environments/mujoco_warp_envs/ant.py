import torch
import warp as wp
from pathlib import Path

from mushroom_rl.environments.mujoco_warp import MuJoCoWarp
from mushroom_rl.environments.mujoco import ObservationType
from mushroom_rl.core.spaces import Box


class AntWarp(MuJoCoWarp):
    """
    Mujoco WARP simulation of the Ant task.

    """

    def __init__(
        self,
        num_envs,
        gamma=0.99,
        horizon=1000,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.5,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        n_substeps=5,
        exclude_current_positions_from_observation=True,
        use_contact_forces=False,
        use_graph_capture=True,
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
            / "ant"
            / "model.xml"
        ).as_posix()

        actuation_spec = [
            "hip_4",
            "ankle_4",
            "hip_1",
            "ankle_1",
            "hip_2",
            "ankle_2",
            "hip_3",
            "ankle_3",
        ]

        observation_spec = [
            ("root_pose", "root", ObservationType.JOINT_POS),
            ("hip_1_pos", "hip_1", ObservationType.JOINT_POS),
            ("ankle_1_pos", "ankle_1", ObservationType.JOINT_POS),
            ("hip_2_pos", "hip_2", ObservationType.JOINT_POS),
            ("ankle_2_pos", "ankle_2", ObservationType.JOINT_POS),
            ("hip_3_pos", "hip_3", ObservationType.JOINT_POS),
            ("ankle_3_pos", "ankle_3", ObservationType.JOINT_POS),
            ("hip_4_pos", "hip_4", ObservationType.JOINT_POS),
            ("ankle_4_pos", "ankle_4", ObservationType.JOINT_POS),
            ("root_vel", "root", ObservationType.JOINT_VEL),
            ("hip_1_vel", "hip_1", ObservationType.JOINT_VEL),
            ("ankle_1_vel", "ankle_1", ObservationType.JOINT_VEL),
            ("hip_2_vel", "hip_2", ObservationType.JOINT_VEL),
            ("ankle_2_vel", "ankle_2", ObservationType.JOINT_VEL),
            ("hip_3_vel", "hip_3", ObservationType.JOINT_VEL),
            ("ankle_3_vel", "ankle_3", ObservationType.JOINT_VEL),
            ("hip_4_vel", "hip_4", ObservationType.JOINT_VEL),
            ("ankle_4_vel", "ankle_4", ObservationType.JOINT_VEL),
        ]

        additional_data_spec = [
            ("torso_pos", "torso", ObservationType.BODY_POS),
            ("torso_vel", "torso", ObservationType.BODY_VEL_WORLD),
        ]

        collision_groups = [
            ("torso", ["torso_geom"]),
            ("floor", ["floor"]),
        ]

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._use_contact_forces = use_contact_forces

        super().__init__(
            num_envs=num_envs,
            xml_file=xml_path,
            gamma=gamma,
            horizon=horizon,
            observation_spec=observation_spec,
            actuation_spec=actuation_spec,
            collision_groups=collision_groups,
            additional_data_spec=additional_data_spec,
            n_substeps=n_substeps,
            use_graph_capture=use_graph_capture,
            nconmax=nconmax,
            njmax=njmax,
            **viewer_params,
        )

    def _modify_mdp_info(self, mdp_info):
        if self._exclude_current_positions_from_observation:
            self.obs_helper.remove_obs("root_pose", 0)
            self.obs_helper.remove_obs("root_pose", 1)
        if self._use_contact_forces:
            self.obs_helper.add_obs("collision_force", 6)
        mdp_info = super()._modify_mdp_info(mdp_info)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _create_observation(self, obs):
        obs = obs.clone()

        if self._use_contact_forces:
            collision_force = self._get_collision_force("torso", "floor")
            obs = torch.cat([obs, collision_force], dim=1)
        return obs

    def _is_finite(self, obs):
        qpos = wp.to_torch(self._data_wp.qpos)  # zero-copy torch view on gpu
        qvel = wp.to_torch(self._data_wp.qvel)
        states = torch.cat([qpos, qvel], dim=1)
        return torch.isfinite(states).all(dim=1)

    def _is_within_z_range(self, obs):
        """Check if Z position of torso is within the healthy range."""
        min_z, max_z = self._healthy_z_range
        z_position = self._read_data("torso_pos")[:, 2]
        return (z_position >= min_z) & (z_position <= max_z)

    def _is_healthy(self, obs):
        return self._is_finite(obs) & self._is_within_z_range(obs)

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

        cost = ctrl_cost
        if self._use_contact_forces:
            collision_force = self.obs_helper.get_from_obs(next_obs, "collision_force")
            lo, hi = self._contact_force_range
            clipped = torch.clamp(collision_force, lo, hi)
            contact_cost = self._contact_cost_weight * (clipped**2).sum(dim=-1)
            cost = cost + contact_cost

        return healthy_r + forward_r - cost

    def setup(self, env_indices, obs):
        """Reset with noise on qpos (uniform) and qvel (gaussian) for the given environments."""
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
            torch.randn(n, self._model.nv, device=device) * self._reset_noise_scale
        )

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

    def get_states(self):
        qpos = wp.to_torch(self._data_wp.qpos)
        qvel = wp.to_torch(self._data_wp.qvel)
        return torch.cat([qpos, qvel], dim=1)
