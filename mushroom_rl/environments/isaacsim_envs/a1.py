from pathlib import Path

import torch

from mushroom_rl.environments.isaacsim_envs.quadruped import QuadrupedIsaac
from mushroom_rl.utils import TorchUtils
from mushroom_rl.utils.isaac_sim import ObservationType


class A1Isaac(QuadrupedIsaac):
    """
    A learning environment for training the A1 quadruped to walk.

    Resembles environment implemented by Rudin et al. for
    "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning"

    """
    def __init__(self, num_envs, horizon, domain_randomization=True, camera_position=(105, 0, 4),
                 camera_target=(95, 0, 0)):
        usd_path = str(Path(__file__).resolve().parent / "robots_usds/a1/a1.usd")
        device = TorchUtils.get_device()

        action_spec = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
        ]
        default_joint_angles = torch.tensor([
            0.1, 0.8, -1.5,
            -0.1, 0.8, -1.5,
            0.1, 1., -1.5,
            -0.1, 1., -1.5
        ], device=device)
        observation_spec = [
            ("base_lin_vel", "", ObservationType.BODY_LIN_VEL, None),
            ("base_ang_vel", "", ObservationType.BODY_ANG_VEL, None),

            ("joint_pos", "", ObservationType.JOINT_POS, action_spec),
            ("joint_vel", "", ObservationType.JOINT_VEL, action_spec)
        ]
        additional_data_spec = [
            ("body_rot", "", ObservationType.BODY_ROT, None),
            ("body_vel", "", ObservationType.BODY_VEL, None)
        ]

        # one collision group is faster, of course it would be cleaner with 3 (feet, body, lower_body)
        collision_groups = [
            ("body", ["/trunk", "/FL_foot", "/FR_foot", "/RL_foot", "/RR_foot",
                      "/FL_thigh", "/FR_thigh", "/RL_thigh", "/RR_thigh",
                      "/FL_calf", "/FR_calf", "/RL_calf", "/RR_calf"]),
        ]
        self._trunk_idx = 0
        self._feet_ids = slice(1, 5)
        self._lower_bodies_ids = slice(5, None)

        super().__init__(usd_path, action_spec, default_joint_angles, observation_spec, additional_data_spec,
                         collision_groups, num_envs, horizon, domain_randomization, camera_position, camera_target)

    def is_absorbing(self, obs):
        trunk_forces = self._collision_helper.get_net_contact_forces("body", dt=self._timestep)[:, self._trunk_idx, :]
        fallen = torch.norm(trunk_forces, dim=-1) > 1.
        return fallen

    def _get_obs_normilization_vec(self):
        v = torch.zeros(self._observation_helper.obs_length, device=TorchUtils.get_device())

        lin_vel = self._observation_helper.obs_idx_map["base_lin_vel"]
        ang_vel = self._observation_helper.obs_idx_map["base_ang_vel"]
        joint_positions = self._observation_helper.obs_idx_map["joint_pos"]
        joint_velocities = self._observation_helper.obs_idx_map["joint_vel"]
        gravity = self._observation_helper.obs_idx_map["projected_gravity"]
        commands = self._observation_helper.obs_idx_map["commands"]
        actions = self._observation_helper.obs_idx_map["actions"]

        v[lin_vel] = 2.0
        v[ang_vel] = 0.25
        v[joint_positions] = 1.00
        v[joint_velocities] = 0.05
        v[gravity] = 1.
        v[commands[0:2]] = 2.0
        v[commands[2]] = 0.25
        v[actions] = 1.

        return v

    def _compute_torque(self, action, joint_vels, joint_pos):
        actions_scaled = action * 0.25
        self._torques = 20.0 * (actions_scaled + self._default_joint_angles - joint_pos) - 0.5 * joint_vels
        self._torques = torch.clip(self._torques, -self._effort_limit, self._effort_limit)

        return self._torques

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        forces = self._collision_helper.get_net_contact_forces("body", dt=self._timestep)[:, self._lower_bodies_ids]
        contact = torch.norm(forces, dim=-1) > 0.1
        return torch.sum(contact, dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self._collision_helper.get_net_contact_forces("body", dt=self._timestep)[:, self._feet_ids, 2] > 1.
        contact_filt = torch.logical_or(contact, self._last_contacts)
        self._last_contacts = contact
        first_contact = (self._feet_air_time > 0.) * contact_filt
        self._feet_air_time += self.dt
        # reward only on first contact with the ground
        rew_air_time = torch.sum((self._feet_air_time - 0.5) * first_contact, dim=1)
        rew_air_time *= torch.norm(self._commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self._feet_air_time *= ~contact_filt
        return rew_air_time
