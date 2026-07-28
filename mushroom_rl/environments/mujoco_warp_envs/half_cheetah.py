import torch
import warp as wp
from pathlib import Path

from mushroom_rl.environments.mujoco_warp import MuJoCoWarp
from mushroom_rl.environments.mujoco import ObservationType
from mushroom_rl.core.spaces import Box


class HalfCheetahWarp(MuJoCoWarp):
    """
    Mujoco WARP simulation of the HalfCheetah task.

    """

    def __init__(
        self,
        num_envs,
        gamma=0.99,
        horizon=1000,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        n_substeps=5,
        exclude_current_positions_from_observation=True,
        use_graph_capture=True,
        warmup_steps=3,
        nconmax=200,
        njmax=200,
        **viewer_params,
    ):
        """Constructor."""
        xml_path = (
            Path(__file__).resolve().parent.parent
            / "mujoco_envs"
            / "data"
            / "half_cheetah"
            / "model.xml"
        ).as_posix()

        actuation_spec = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]

        observation_spec = [
            ("z_pos", "rootz", ObservationType.JOINT_POS),
            ("y_pos", "rooty", ObservationType.JOINT_POS),
            ("bthigh_pos", "bthigh", ObservationType.JOINT_POS),
            ("bshin_pos", "bshin", ObservationType.JOINT_POS),
            ("bfoot_pos", "bfoot", ObservationType.JOINT_POS),
            ("fthigh_pos", "fthigh", ObservationType.JOINT_POS),
            ("fshin_pos", "fshin", ObservationType.JOINT_POS),
            ("ffoot_pos", "ffoot", ObservationType.JOINT_POS),
            ("x_vel", "rootx", ObservationType.JOINT_VEL),
            ("z_vel", "rootz", ObservationType.JOINT_VEL),
            ("y_vel", "rooty", ObservationType.JOINT_VEL),
            ("bthigh_vel", "bthigh", ObservationType.JOINT_VEL),
            ("bshin_vel", "bshin", ObservationType.JOINT_VEL),
            ("bfoot_vel", "bfoot", ObservationType.JOINT_VEL),
            ("fthigh_vel", "fthigh", ObservationType.JOINT_VEL),
            ("fshin_vel", "fshin", ObservationType.JOINT_VEL),
            ("ffoot_vel", "ffoot", ObservationType.JOINT_VEL),
        ]

        additional_data_spec = [
            ("x_pos", "rootx", ObservationType.JOINT_POS),
            ("torso_vel", "torso", ObservationType.BODY_VEL_WORLD),
        ]

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
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
            use_graph_capture=use_graph_capture,
            warmup_steps=warmup_steps,
            nconmax=nconmax,
            njmax=njmax,
            **viewer_params,
        )

    def _modify_mdp_info(self, mdp_info):
        if not self._exclude_current_positions_from_observation:
            self.obs_helper.add_obs("x_pos", 1)
        mdp_info = super()._modify_mdp_info(mdp_info)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _create_observation(self, obs):
        obs = obs.clone()

        if not self._exclude_current_positions_from_observation:
            x_pos = self._read_data("x_pos")
            obs = torch.cat([obs, x_pos], dim=1)
        return obs

    def is_absorbing(self, obs):
        return torch.zeros(self._num_envs, dtype=torch.bool, device=obs.device)

    def reward(self, obs, action, next_obs, absorbing):
        torso_vel = self._read_data("torso_vel")
        forward_r = self._forward_reward_weight * torso_vel[:, 3]

        action_t = torch.as_tensor(
            action, dtype=forward_r.dtype, device=forward_r.device
        )
        ctrl_cost = self._ctrl_cost_weight * (action_t**2).sum(dim=-1)

        return forward_r - ctrl_cost

    def setup(self, env_indices, obs):
        """Reset with small uniform noise on qpos and qvel for the given environments."""
        super().setup(env_indices, obs)

        qpos = wp.to_torch(self._data_wp.qpos)
        qvel = wp.to_torch(self._data_wp.qvel)

        device = qpos.device
        idx = (
            torch.as_tensor(env_indices, device=device, dtype=torch.long)
            if not isinstance(env_indices, torch.Tensor)
            else env_indices.to(device).long()
        )

        n = idx.shape[0]
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
        torso_vel = self._read_data("torso_vel")
        forward_r = self._forward_reward_weight * torso_vel[:, 3]
        return {
            "forward_reward": forward_r,
        }
