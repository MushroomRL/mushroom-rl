from isaacsim.core.utils.torch.maths import torch_rand_float

from pathlib import Path

import torch

from mushroom_rl.environments.isaacsim_envs.quadruped import QuadrupedIsaac
from mushroom_rl.utils import TorchUtils
from mushroom_rl.utils.isaac_sim import ObservationType


class HoneyBadgerIsaac(QuadrupedIsaac):
    """
    A learning environment for training the Honey Badger quadroped to walk.
    Honey Badger is a Robot from MAB Robotics: https://www.mabrobotics.pl/
    """
    MAX_NR_DELAY_STEPS = 1
    MIXED_CHANCE = 0.05

    def __init__(self, num_envs, horizon, domain_randomization=True, camera_pos=(105, 0, 4),
                 camera_target=(95, 0, 0)):
        usd_path, action_spec, default_joint_angles, default_joint_max_vel, sub_bodies, collision_groups = \
            self._robot_config()
        self._default_joint_max_vel = default_joint_max_vel

        observation_spec = [
            ("base_lin_vel", "", ObservationType.BODY_LIN_VEL, None),
            ("base_ang_vel", "", ObservationType.BODY_ANG_VEL, None),
            ("joint_pos", "", ObservationType.JOINT_POS, action_spec),
            ("joint_vel", "", ObservationType.JOINT_VEL, action_spec),
            ("base_pos", "", ObservationType.BODY_POS, None),
        ]
        additional_data_spec = [
            ("body_rot", "", ObservationType.BODY_ROT, None),
            ("body_vel", "", ObservationType.BODY_VEL, None),
            ("trunk_mass", "", ObservationType.SUB_BODY_MASS, "body"),
            ("trunk_inertia", "", ObservationType.SUB_BODY_INERTIA, "body"),
            ("trunk_com", "", ObservationType.SUB_BODY_COM_POS, "body"),
            ("FL_foot_scale", "/fl_foot", ObservationType.BODY_SCALE, None),
            ("FR_foot_scale", "/fr_foot", ObservationType.BODY_SCALE, None),
            ("RL_foot_scale", "/rl_foot", ObservationType.BODY_SCALE, None),
            ("RR_foot_scale", "/rr_foot", ObservationType.BODY_SCALE, None),
            ("torque_limit", "", ObservationType.JOINT_MAX_EFFORT, action_spec),
            ("max_joint_vel", "", ObservationType.JOINT_MAX_VELOCITY, action_spec),
            ("joint_range", "", ObservationType.JOINT_MAX_POS, action_spec),
            ("joint_armature", "", ObservationType.JOINT_ARMATURES, action_spec),
            ("joint_frictionloss", "", ObservationType.JOINT_FRICTION_STATIC, action_spec),
            ("joint_damping", "", ObservationType.JOINT_GAIN_DAMPING, action_spec),
            ("joint_stiffness", "", ObservationType.JOINT_GAIN_STIFFNESS, action_spec),
            ("joint_default_pos", "", ObservationType.JOINT_DEFAULT_POS, action_spec),
            ("robot_mass", "", ObservationType.SUB_BODY_MASS, sub_bodies),
        ]

        super().__init__(usd_path, action_spec, default_joint_angles, observation_spec, additional_data_spec,
                         collision_groups, num_envs, horizon, domain_randomization, camera_pos, camera_target,
                         reward_weights=dict(torques=-0.0001, height=-4.0))

        # domain randomization
        self._current_mixed = False
        self._current_nr_delay_steps = 0
        self._action_history = torch.zeros((self.MAX_NR_DELAY_STEPS + 1, self.number, len(self._action_spec)),
                                           device=TorchUtils.get_device())

    def setup(self, env_indices, obs):
        super().setup(env_indices, obs)
        self._action_history[:, env_indices, :] = 0

    def is_absorbing(self, obs):
        forces = self._collision_helper.get_net_contact_forces("body", dt=self._timestep)
        fallen = torch.any(torch.norm(forces, dim=-1) > 0., dim=-1)
        return fallen

    # domain randomization -------------------------------------------------------------------------------------

    def sample_unseen_noise_factors(
            self, env_indices,
            trunk_mass_factor=0.25,
            trunk_com_factor=0.25,
            foot_size_factor=0.03,
            joint_damping_factor=0.5,
            joint_armature_factor=0.5,
            joint_stiffness_factor=0.5,
            joint_friction_factor=0.5,
            motor_strength_factor=0.25,
            p_gain_factor=0.25,
            d_gain_factor=0.25,
            position_offset=0.05):
        n_envs = env_indices.shape[0]

        self._nf_trunk_mass[env_indices] = torch_rand_float(
            1 - trunk_mass_factor, 1 + trunk_mass_factor, (n_envs, 1), TorchUtils.get_device()
        )
        self._nf_trunk_com[env_indices] = torch_rand_float(
            1 - trunk_com_factor, 1 + trunk_com_factor, (n_envs, 1), TorchUtils.get_device()
        )
        self._nf_foot_size[env_indices] = torch_rand_float(
            1 - foot_size_factor, 1 + foot_size_factor, (n_envs, 1), TorchUtils.get_device()
        )
        self._nf_joint_damping[env_indices] = torch_rand_float(
            1 - joint_damping_factor, 1 + joint_damping_factor, (n_envs, 1), TorchUtils.get_device()
        )
        self._nf_joint_stiffness[env_indices] = torch_rand_float(
            1 - joint_stiffness_factor, 1 + joint_stiffness_factor, (n_envs, 1), TorchUtils.get_device()
        )
        self._nf_joint_armature[env_indices] = torch_rand_float(
            1 - joint_armature_factor, 1 + joint_armature_factor, (n_envs, 1), TorchUtils.get_device()
        )
        self._nf_joint_friction[env_indices] = torch_rand_float(
            1 - joint_friction_factor, 1 + joint_friction_factor, (n_envs, 1), TorchUtils.get_device()
        )

        # control function
        self._nf_p_gain[env_indices] = torch_rand_float(
            1 - p_gain_factor, 1 + p_gain_factor, (n_envs, 1), TorchUtils.get_device()
        )
        self._nf_d_gain[env_indices] = torch_rand_float(
            1 - d_gain_factor, 1 + d_gain_factor, (n_envs, 1), TorchUtils.get_device()
        )
        self._nf_motor_strength[env_indices] = torch_rand_float(
            1 - motor_strength_factor, 1 + motor_strength_factor, (n_envs, 1), TorchUtils.get_device()
        )
        self._joint_position_offset[env_indices] = torch_rand_float(
            -position_offset, position_offset, (n_envs, len(self._action_spec)), TorchUtils.get_device()
        )

    def sample_seen_parameters(
            self, env_indices,
            stay_at_default_percentage=0.3,
            add_trunk_mass_min=-0.8, add_trunk_mass_max=0.8,
            add_com_displacement_min=-0.0025, add_com_displacement_max=0.0025,
            foot_scaling_min=0.975, foot_scaling_max=1.025,
            torque_limit_factor=0.3,
            add_joint_nominal_position_min=-0.01, add_joint_nominal_position_max=0.01,
            joint_velocity_factor=0.15,
            add_joint_range_min=-0.05, add_joint_range_max=0.05,
            joint_damping_min=0.0, joint_damping_max=0.3,
            joint_armature_min=0.009, joint_armature_max=0.023,
            joint_stiffness_min=0.0, joint_stiffness_max=0.5,
            joint_friction_loss_min=0.0, joint_friction_loss_max=0.1,
            add_p_gain_min=-3.0, add_p_gain_max=3.0,
            add_d_gain_min=-0.1, add_d_gain_max=0.1,
            add_scaling_factor_min=-0.03, add_scaling_factor_max=0.03,):
        n_envs = env_indices.shape[0]
        n_joints = len(self._action_spec)

        # trunk mass
        trunk_mass = self._default_trunk_mass \
            + torch_rand_float(add_trunk_mass_min, add_trunk_mass_max, (n_envs, 1), TorchUtils.get_device())
        actual_trunk_mass = trunk_mass * self._nf_trunk_mass[env_indices]
        self._observation_helper.write_data("trunk_mass", actual_trunk_mass, env_indices, True)
        actual_trunk_inertia = self._default_trunk_inertia + (actual_trunk_mass / self._default_trunk_mass)
        self._observation_helper.write_data("trunk_inertia", actual_trunk_inertia.unsqueeze(1), env_indices, True)
        self._seen_mass[env_indices, 0] = trunk_mass.squeeze(1)
        self._seen_summed_mass = torch.sum(self._seen_mass, dim=1)

        # trunk com
        actual_trunk_com = self._default_trunk_com \
            + torch_rand_float(add_com_displacement_min, add_com_displacement_max, (n_envs, 1), TorchUtils.get_device())
        self._seen_trunk_com[env_indices] = actual_trunk_com.unsqueeze(1)
        actual_trunk_com *= self._nf_trunk_com[env_indices]
        self._observation_helper.write_data("trunk_com", actual_trunk_com.unsqueeze(1), env_indices, True)

        # foot scaling
        self._seen_foot_scaling = torch_rand_float(foot_scaling_min, foot_scaling_max, (n_envs, 4),
                                                   TorchUtils.get_device())
        actual_foot_scaling = self._seen_foot_scaling * self._nf_foot_size[env_indices]
        for i, name in enumerate(["FL_foot_scale", "FR_foot_scale", "RL_foot_scale", "RR_foot_scale"]):
            scale = actual_foot_scaling[env_indices, i].unsqueeze(1).repeat(1, 3)
            self._observation_helper.write_data(name, scale, env_indices, True)

        # joint nominal position
        self._seen_joint_nominal_pos[env_indices] = self._default_joint_nominal_pos \
            + torch_rand_float(
                add_joint_nominal_position_min, add_joint_nominal_position_max, (n_envs, n_joints),
                TorchUtils.get_device()
            )

        # joint torque limit
        self._seen_torque_limit[env_indices] = self._default_torque_limit \
            * (1 + torch_rand_float(
                -torque_limit_factor, torque_limit_factor, (n_envs, n_joints), TorchUtils.get_device()
            ))
        self._observation_helper.write_data("torque_limit", self._seen_torque_limit[env_indices], env_indices, True)

        # joint max velocity
        self._seen_joint_max_vel[env_indices] = self._default_joint_max_vel \
            * (1 + torch_rand_float(
                -joint_velocity_factor, joint_velocity_factor, (n_envs, n_joints), TorchUtils.get_device()
            ))
        self._observation_helper.write_data("max_joint_vel", self._seen_joint_max_vel[env_indices], env_indices, True)

        # joint damping, stiffness, armature, frictionloss
        stay_at_default_mask = torch_rand_float(0, 1, (n_envs, 1), TorchUtils.get_device()) < stay_at_default_percentage
        stay_at_default_mask = stay_at_default_mask.squeeze()
        stay_at_default_idx = env_indices[stay_at_default_mask]
        self._seen_joint_damping[stay_at_default_idx] = self._default_joint_damping
        self._seen_joint_stiffness[stay_at_default_idx] = self._default_joint_stiffness
        self._seen_joint_armature[stay_at_default_idx] = self._default_joint_armature
        self._seen_joint_frictionloss[stay_at_default_idx] = self._default_joint_frictionloss

        not_stay_at_default_mask = torch.logical_not(stay_at_default_mask)
        not_stay_at_default_idx = env_indices[not_stay_at_default_mask]
        num_envs_not_default = not_stay_at_default_idx.shape[0]
        self._seen_joint_damping[not_stay_at_default_idx] = torch_rand_float(
            joint_damping_min, joint_damping_max, (num_envs_not_default, n_joints), TorchUtils.get_device()
        )
        self._seen_joint_stiffness[not_stay_at_default_idx] = torch_rand_float(
            joint_stiffness_min, joint_stiffness_max, (num_envs_not_default, n_joints), TorchUtils.get_device()
        )
        self._seen_joint_armature[not_stay_at_default_idx] = torch_rand_float(
            joint_armature_min, joint_armature_max, (num_envs_not_default, n_joints), TorchUtils.get_device()
        )
        self._seen_joint_frictionloss[not_stay_at_default_idx] = torch_rand_float(
            joint_friction_loss_min, joint_friction_loss_max, (num_envs_not_default, n_joints),
            TorchUtils.get_device()
        )

        # chceck if damping is difference in scale
        self._observation_helper.write_data(
            "joint_damping", self._seen_joint_damping[env_indices] * self._nf_joint_damping[env_indices],
            env_indices, True)
        self._observation_helper.write_data(
            "joint_stiffness", self._seen_joint_stiffness[env_indices] * self._nf_joint_stiffness[env_indices],
            env_indices, True)
        self._observation_helper.write_data(
            "joint_armature", self._seen_joint_armature[env_indices] * self._nf_joint_armature[env_indices],
            env_indices, True)
        self._observation_helper.write_data(
            "joint_frictionloss", self._seen_joint_frictionloss[env_indices] * self._nf_joint_friction[env_indices],
            env_indices, True)

        # used for control function
        self._seen_p_gain[env_indices] = 20 + torch_rand_float(
            add_p_gain_min, add_p_gain_max, (n_envs, n_joints), TorchUtils.get_device()
        )
        self._seen_d_gain[env_indices] = 0.5 + torch_rand_float(
            add_d_gain_min, add_d_gain_max, (n_envs, n_joints), TorchUtils.get_device()
        )
        self._seen_scaling_factor[env_indices] = 0.25 + torch_rand_float(
            add_scaling_factor_min, add_scaling_factor_max, (n_envs, n_joints), TorchUtils.get_device()
        )

        self._unseen_p_gain[env_indices] = self._seen_p_gain[env_indices] * self._nf_p_gain[env_indices]
        self._unseen_d_gain[env_indices] = self._seen_d_gain[env_indices] * self._nf_d_gain[env_indices]

    def add_domain_randomization_observations(
            self,
            stay_at_default_percentage=0.3,
            add_trunk_mass_min=-0.8, add_trunk_mass_max=0.8,
            add_com_displacement_min=-0.0025, add_com_displacement_max=0.0025,
            foot_scaling_min=0.975, foot_scaling_max=1.025,
            torque_limit_factor=0.3,
            add_joint_nominal_position_min=-0.01, add_joint_nominal_position_max=0.01,
            joint_velocity_factor=0.15,
            add_joint_range_min=-0.05, add_joint_range_max=0.05,
            joint_damping_min=0.0, joint_damping_max=0.3,
            joint_armature_min=0.009, joint_armature_max=0.023,
            joint_stiffness_min=0.0, joint_stiffness_max=0.5,
            joint_friction_loss_min=0.0, joint_friction_loss_max=1.0,
            add_p_gain_min=-3.0, add_p_gain_max=3.0,
            add_d_gain_min=-0.1, add_d_gain_max=0.1,
            add_scaling_factor_min=-0.03, add_scaling_factor_max=0.03,):
        n_joints = len(self._action_spec)

        # joints
        self._observation_helper.add_obs(
            name="joint_nominal_position",
            length=n_joints,
            min_value=(self._default_joint_angles + add_joint_nominal_position_min) / 4.6,
            max_value=(self._default_joint_angles + add_joint_nominal_position_max) / 4.6
        )
        self._observation_helper.add_obs(
            name="torque_limit",
            length=n_joints,
            min_value=(self._default_torque_limit * (1 - torque_limit_factor)) / (1000.0 / 2) - 1.0,
            max_value=(self._default_torque_limit * (1. + torque_limit_factor)) / (1000.0 / 2) - 1.0
        )
        self._observation_helper.add_obs(
            name="joint_max_velocity",
            length=n_joints,
            min_value=(self._default_joint_max_vel * (1 - joint_velocity_factor)) / (35.0 / 2) - 1.0,
            max_value=(self._default_joint_max_vel * (1 + joint_velocity_factor)) / (35.0 / 2) - 1.0
        )
        self._observation_helper.add_obs(
            name="joint_damping",
            length=n_joints,
            min_value=joint_damping_min / (10.0 / 2) - 1.0,
            max_value=joint_damping_max / (10.0 / 2) - 1.0
        )
        self._observation_helper.add_obs(
            name="joint_stiffness",
            length=n_joints,
            min_value=joint_stiffness_min / (30.0 / 2) - 1.0,
            max_value=joint_stiffness_max / (30.0 / 2) - 1.0
        )
        self._observation_helper.add_obs(
            name="joint_armature",
            length=n_joints,
            min_value=joint_armature_min / (0.2 / 2) - 1.0,
            max_value=joint_armature_max / (0.2 / 2) - 1.0
        )
        self._observation_helper.add_obs(
            name="joint_frictionloss",
            length=n_joints,
            min_value=joint_friction_loss_min / (1.2 / 2) - 1.0,
            max_value=joint_friction_loss_max / (1.2 / 2) - 1.0
        )
        self._observation_helper.add_obs(
            name="p_gain",
            length=n_joints,
            min_value=20 + add_p_gain_min / (100.0 / 2) - 1.0,
            max_value=20 + add_p_gain_max / (100.0 / 2) - 1.0
        )
        self._observation_helper.add_obs(
            name="d_gain",
            length=n_joints,
            min_value=0.5 + add_d_gain_min / (2.0 / 2) - 1.0,
            max_value=0.5 + add_d_gain_max / (2.0 / 2) - 1.0
        )
        self._observation_helper.add_obs(
            name="action_scaling_factor",
            length=n_joints,
            min_value=0.25 + add_scaling_factor_min / (0.8 / 2) - 1.0,
            max_value=0.25 + add_scaling_factor_max / (0.8 / 2) - 1.0
        )

        # mass, com foot scaling
        self._observation_helper.add_obs(
            name="mass",
            length=1,
            min_value=-torch.inf,
            max_value=torch.inf
        )

    def delay_action(self, action):
        if self._current_mixed:
            self._current_nr_delay_steps = torch.randint(0, self.MAX_NR_DELAY_STEPS + 1, (1,)).item()

        self._action_history = torch.roll(self._action_history, -1, dims=0)
        self._action_history[-1] = action

        chosen_action = self._action_history[-1 - self._current_nr_delay_steps]

        return chosen_action

    # construction-time hooks -------------------------------------------------------------------------------------

    def _register_domain_randomization_observations(self):
        # establishes the nominal per-joint control parameters (_seen_*/_unseen_*/_nf_* below) the control law
        # in _compute_torque always relies on; domain randomization only perturbs them further, so this has to
        # run regardless of whether it is enabled
        self._init_domain_randomization_parameters()

        if self._domain_randomization:
            self.add_domain_randomization_observations()

    def _get_obs_normilization_vec(self):
        v = torch.ones((self._observation_helper.obs_length), device=TorchUtils.get_device())

        lin_vel = self._observation_helper.obs_idx_map["base_lin_vel"]
        ang_vel = self._observation_helper.obs_idx_map["base_ang_vel"]
        joint_positions = self._observation_helper.obs_idx_map["joint_pos"]
        joint_velocities = self._observation_helper.obs_idx_map["joint_vel"]
        gravity = self._observation_helper.obs_idx_map["projected_gravity"]
        commands = self._observation_helper.obs_idx_map["commands"]
        actions = self._observation_helper.obs_idx_map["actions"]
        pos = self._observation_helper.obs_idx_map["base_pos"]

        v[lin_vel] = 2.0
        v[ang_vel] = 0.25
        v[joint_positions] = 1.00
        v[joint_velocities] = 0.05
        v[gravity] = 1.
        v[commands[0:2]] = 2.0
        v[commands[2]] = 0.25
        v[actions] = 1.
        v[pos] = 1 / 0.4

        if self._domain_randomization:
            joint_nominal_pos_ids = self._observation_helper.obs_idx_map["joint_nominal_position"]
            torque_limit_ids = self._observation_helper.obs_idx_map["torque_limit"]
            joint_max_velocity_ids = self._observation_helper.obs_idx_map["joint_max_velocity"]
            joint_damping_ids = self._observation_helper.obs_idx_map["joint_damping"]
            joint_stiffness_ids = self._observation_helper.obs_idx_map["joint_stiffness"]
            joint_armature_ids = self._observation_helper.obs_idx_map["joint_armature"]
            joint_frictionloss_ids = self._observation_helper.obs_idx_map["joint_frictionloss"]
            p_gain_ids = self._observation_helper.obs_idx_map["p_gain"]
            d_gain_ids = self._observation_helper.obs_idx_map["d_gain"]
            action_scaling_factor_ids = self._observation_helper.obs_idx_map["action_scaling_factor"]
            mass_ids = self._observation_helper.obs_idx_map["mass"]

            v[joint_nominal_pos_ids] = 1. / 4.6
            v[torque_limit_ids] = 1. / (1000.0 / 2)
            v[joint_max_velocity_ids] = 1. / (35.0 / 2)
            v[joint_damping_ids] = 1. / (10.0 / 2)
            v[joint_stiffness_ids] = 1. / (30.0 / 2)
            v[joint_armature_ids] = 1. / (0.2 / 2)
            v[joint_frictionloss_ids] = 1. / (1.2 / 2)
            v[p_gain_ids] = 1. / (100.0 / 2)
            v[d_gain_ids] = 1. / (2.0 / 2)
            v[action_scaling_factor_ids] = 1. / (0.8 / 2)
            v[mass_ids] = 1. / (170.0 / 2)

        return v

    # per-step hooks --------------------------------------------------------------------------------------------

    def _sample_setup_joint_pos(self, env_indices):
        if self._domain_randomization:
            return self._seen_joint_nominal_pos[env_indices]
        return super()._sample_setup_joint_pos(env_indices)

    def _push_domain_randomization(self, env_indices):
        if self._domain_randomization:
            do_push = torch_rand_float(0., 1., (len(env_indices), 1),
                                       device=TorchUtils.get_device()).squeeze(-1) < (1. / 750.)
            do_push_ids = env_indices[do_push]
            do_push_ids = do_push_ids[self._episode_length[do_push_ids] > 50]
            self._push_robots(do_push_ids)

            if torch.rand(()).item() < 0.002:
                self._current_mixed = torch.rand(()).item() < self.MIXED_CHANCE
                self._current_nr_delay_steps = 0

            if torch.rand(()).item() < 0.0004:
                self.sample_unseen_noise_factors(torch.arange(0, self.number, 1, device=TorchUtils.get_device()))
                self.sample_seen_parameters(torch.arange(0, self.number, 1, device=TorchUtils.get_device()))

    # observations ------------------------------------------------------------------------------------------------

    def _create_observation(self, obs):
        obs = super()._create_observation(obs)

        base_pos_indices = self._observation_helper.obs_idx_map["base_pos"]
        obs[:, base_pos_indices[:2]] = 0

        return obs

    def _add_domain_randomization_observations(self, obs):
        if self._domain_randomization:
            return self._add_seen_parameters(obs)
        return obs

    # control -----------------------------------------------------------------------------------------------------

    def _apply_action_delay(self, action):
        if self._domain_randomization:
            return self.delay_action(action)
        return action

    def _compute_torque(self, action, joint_vels, joint_pos):
        action_scaled = action * self._seen_scaling_factor
        target_joint_pos = self._seen_joint_nominal_pos + action_scaled

        self._torques = self._unseen_p_gain * (target_joint_pos - joint_pos + self._joint_position_offset) \
            - self._unseen_d_gain * joint_vels
        self._torques *= self._nf_motor_strength
        self._torques = torch.clip(self._torques, -self._seen_torque_limit, self._seen_torque_limit)

        return self._torques

    # reward function -----------------------------------------------------------------------------------------

    def _extra_reward_terms(self, next_obs):
        base_pos = self._observation_helper.get_from_obs(next_obs, "base_pos")
        base_pos_z = base_pos[:, 2]
        return self._reward_height(base_pos_z) * self._reward_weights["height"] * self.dt

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        forces = self._collision_helper.get_net_contact_forces("lower_body", dt=self._timestep)
        contact = torch.norm(forces, dim=-1) > 0.1
        return torch.sum(contact, dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self._collision_helper.get_net_contact_forces("feet", dt=self._timestep)[:, :, 2] > 1.
        contact_filt = torch.logical_or(contact, self._last_contacts)
        self._last_contacts = contact
        first_contact = (self._feet_air_time > 0.) * contact_filt
        self._feet_air_time += self.dt
        # reward only on first contact with the ground
        rew_airTime = torch.sum((self._feet_air_time - 0.5) * first_contact, dim=1)
        rew_airTime *= torch.norm(self._commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self._feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_height(self, base_z):
        # nominal_base_z = 0.316
        nominal_base_z = 0.31
        return torch.square(base_z - nominal_base_z)

    # utilities -----------------------------------------------------------------------------------------------

    def _init_domain_randomization_parameters(self):
        n_envs = self._num_envs
        n_joints = len(self._action_spec)

        # init some seen parameters
        self._seen_joint_damping = self._observation_helper.read_data("joint_damping")
        self._seen_joint_stiffness = self._observation_helper.read_data("joint_stiffness")
        self._seen_joint_armature = self._observation_helper.read_data("joint_armature")
        self._seen_joint_frictionloss = self._observation_helper.read_data("joint_frictionloss")

        self._seen_mass = self._observation_helper.read_data("robot_mass")
        self._seen_summed_mass = torch.sum(self._seen_mass, dim=1)
        self._seen_torque_limit = self._observation_helper.read_data("torque_limit")
        self._seen_joint_nominal_pos = self._default_joint_angles.repeat((n_envs, 1))
        self._seen_joint_max_vel = self._default_joint_max_vel.repeat((n_envs, 1))
        self._seen_foot_scaling = torch.ones((n_envs, 4), device=TorchUtils.get_device())
        self._seen_trunk_com = self._observation_helper.read_data("trunk_com")

        self._observation_helper.write_data("max_joint_vel", self._seen_joint_max_vel, reapply_after_reset=True)

        self._seen_p_gain = torch.full((n_envs, n_joints), 20., device=TorchUtils.get_device())
        self._seen_d_gain = torch.full((n_envs, n_joints), 0.5, device=TorchUtils.get_device())
        self._seen_scaling_factor = torch.full((n_envs, n_joints), 0.25, device=TorchUtils.get_device())

        self._unseen_p_gain = torch.full((n_envs, n_joints), 20., device=TorchUtils.get_device())
        self._unseen_d_gain = torch.full((n_envs, n_joints), 0.5, device=TorchUtils.get_device())

        self._default_trunk_mass = self._observation_helper.read_data("trunk_mass")[0].clone().detach()
        self._default_trunk_inertia = self._observation_helper.read_data("trunk_inertia")[0].clone().detach()
        self._default_trunk_com = self._observation_helper.read_data("trunk_com")[0].clone().detach()
        self._default_torque_limit = self._seen_torque_limit[0].clone().detach()
        self._default_joint_nominal_pos = self._seen_joint_nominal_pos[0].clone().detach()
        self._default_joint_range = self._observation_helper.read_data("joint_range")[0].clone().detach()
        self._default_joint_damping = self._seen_joint_damping[0].clone().detach()
        self._default_joint_stiffness = self._seen_joint_stiffness[0].clone().detach()
        self._default_joint_armature = self._seen_joint_armature[0].clone().detach()
        self._default_joint_frictionloss = self._seen_joint_frictionloss[0].clone().detach()

        self._nf_trunk_mass = torch.ones((n_envs, 1), device=TorchUtils.get_device())
        self._nf_trunk_com = torch.ones((n_envs, 1), device=TorchUtils.get_device())
        self._nf_foot_size = torch.ones((n_envs, 1), device=TorchUtils.get_device())
        self._nf_joint_damping = torch.ones((n_envs, 1), device=TorchUtils.get_device())
        self._nf_joint_stiffness = torch.ones((n_envs, 1), device=TorchUtils.get_device())
        self._nf_joint_armature = torch.ones((n_envs, 1), device=TorchUtils.get_device())
        self._nf_joint_friction = torch.ones((n_envs, 1), device=TorchUtils.get_device())

        self._nf_p_gain = torch.ones((n_envs, 1), device=TorchUtils.get_device())
        self._nf_d_gain = torch.ones((n_envs, 1), device=TorchUtils.get_device())
        self._nf_motor_strength = torch.ones((n_envs, 1), device=TorchUtils.get_device())
        self._joint_position_offset = torch.zeros((n_envs, n_joints), device=TorchUtils.get_device())

    def _add_seen_parameters(self, obs):
        joint_nominal_pos_ids = self._observation_helper.obs_idx_map["joint_nominal_position"]
        obs[:, joint_nominal_pos_ids] = self._seen_joint_nominal_pos

        torque_limit_ids = self._observation_helper.obs_idx_map["torque_limit"]
        obs[:, torque_limit_ids] = self._seen_torque_limit

        joint_max_velocity_ids = self._observation_helper.obs_idx_map["joint_max_velocity"]
        obs[:, joint_max_velocity_ids] = self._seen_joint_max_vel

        joint_damping_ids = self._observation_helper.obs_idx_map["joint_damping"]
        obs[:, joint_damping_ids] = self._seen_joint_damping

        joint_stiffness_ids = self._observation_helper.obs_idx_map["joint_stiffness"]
        obs[:, joint_stiffness_ids] = self._seen_joint_stiffness

        joint_armature_ids = self._observation_helper.obs_idx_map["joint_armature"]
        obs[:, joint_armature_ids] = self._seen_joint_armature

        joint_frictionloss_ids = self._observation_helper.obs_idx_map["joint_frictionloss"]
        obs[:, joint_frictionloss_ids] = self._seen_joint_frictionloss

        p_gain_ids = self._observation_helper.obs_idx_map["p_gain"]
        obs[:, p_gain_ids] = self._seen_p_gain

        d_gain_ids = self._observation_helper.obs_idx_map["d_gain"]
        obs[:, d_gain_ids] = self._seen_d_gain

        action_scaling_factor_ids = self._observation_helper.obs_idx_map["action_scaling_factor"]
        obs[:, action_scaling_factor_ids] = self._seen_scaling_factor

        mass_ids = self._observation_helper.obs_idx_map["mass"]
        obs[:, mass_ids] = self._seen_summed_mass.unsqueeze(1)

        return obs

    @staticmethod
    def _robot_config():
        """
        Returns:
            The robot-specific configuration a :class:`HoneyBadgerIsaac` family member is built from, as the
            tuple ``(usd_path, action_spec, default_joint_angles, default_joint_max_vel, sub_bodies,
            collision_groups)``. Overridden by subclasses simulating a different robot of the same family.

        """
        usd_path = str(Path(__file__).resolve().parent / "robots_usds/honey_badger/honey_badger.usd")
        action_spec = [
            "fl_j0", "fl_j1", "fl_j2",
            "fr_j0", "fr_j1", "fr_j2",
            "rl_j0", "rl_j1", "rl_j2",
            "rr_j0", "rr_j1", "rr_j2"
        ]
        device = TorchUtils.get_device()
        default_joint_angles = torch.tensor([
            0.1, -0.8, 1.5,
            -0.1, 0.8, -1.5,
            0.1, -1., 1.5,
            -0.1, 1., -1.5
        ], device=device)
        default_joint_max_vel = torch.tensor([25.] * 12, device=device)
        sub_bodies = [
            "body",
            "fl_l0", "fr_l0", "rl_l0", "rr_l0",
            "fl_l1", "fr_l1", "rl_l1", "rr_l1",
            "fl_l2", "fr_l2", "rl_l2", "rr_l2",
            "fl_foot", "fr_foot", "rl_foot", "rr_foot"
        ]
        collision_groups = [
            ("feet", ["/fl_foot", "/fr_foot", "/rl_foot", "/rr_foot"]),
            ("body", ["/body", "/fl_l1", "/fr_l1", "/rl_l1", "/rr_l1"]),
            ("lower_body", ["/fl_l2", "/fr_l2", "/rl_l2", "/rr_l2"])
        ]
        return usd_path, action_spec, default_joint_angles, default_joint_max_vel, sub_bodies, collision_groups
