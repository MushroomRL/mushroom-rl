from mushroom_rl.environments import IsaacSim
from mushroom_rl.utils.isaac_sim import ObservationType, ActionType
from mushroom_rl.rl_utils.spaces import Box

import numpy as np
import torch
from pathlib import Path

class HoneyBadgerWalking(IsaacSim):
    """
    A learning environment for training the Honey Badger quadroped to walk. 
    Honey Badger is a Robot from MAB Robotics: https://www.mabrobotics.pl/
    """
    MAX_NR_DELAY_STEPS = 1
    MIXED_CHANCE = 0.05

    def __init__(self, num_envs, horizon, headless, domain_randomization=True, camera_pos=(105, 0, 4), camera_target=(95, 0, 0)):
        usd_path = str(Path(__file__).resolve().parent / "robots_usds/honey_badger/honey_badger.usd")
        self.NUM_JOINTS = 12

        backend="torch"
        device="cuda:0"

        self.domain_randomization = domain_randomization

        self._action_spec = [
            "fl_j0", "fl_j1", "fl_j2",
            "fr_j0", "fr_j1", "fr_j2",   
            "rl_j0", "rl_j1", "rl_j2", 
            "rr_j0", "rr_j1", "rr_j2"
        ]
        self._default_joint_angles = torch.tensor([
            0.1, -0.8, 1.5,
            -0.1, 0.8, -1.5,
            0.1, -1., 1.5,
            -0.1, 1., -1.5
        ], device=device)
        self._default_joint_max_vel = torch.tensor([
            25., 25., 25.,
            25., 25., 25.,
            25., 25., 25.,
            25., 25., 25.
        ],device=device)
        
        observation_spec = [
            ("base_lin_vel", "", ObservationType.BODY_LIN_VEL, None),
            ("base_ang_vel", "", ObservationType.BODY_ANG_VEL, None),
            ("joint_pos", "", ObservationType.JOINT_POS, self._action_spec),
            ("joint_vel", "", ObservationType.JOINT_VEL, self._action_spec),
            ("base_pos", "", ObservationType.BODY_POS, None),
        ]
        sub_bodies = [
            "body", 
            "fl_l0", "fr_l0", "rl_l0", "rr_l0", 
            "fl_l1", "fr_l1", "rl_l1", "rr_l1", 
            "fl_l2", "fr_l2", "rl_l2", "rr_l2",
            "fl_foot", "fr_foot", "rl_foot", "rr_foot"
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
            ("torque_limit", "", ObservationType.JOINT_MAX_EFFORT, self._action_spec),
            ("max_joint_vel", "", ObservationType.JOINT_MAX_VELOCITY, self._action_spec),
            ("joint_range", "", ObservationType.JOINT_MAX_POS, self._action_spec),
            ("joint_armature", "", ObservationType.JOINT_ARMATURES, self._action_spec),
            ("joint_frictionloss", "", ObservationType.JOINT_FRICTION, self._action_spec),
            ("joint_damping", "", ObservationType.JOINT_GAIN_DAMPING, self._action_spec),
            ("joint_stiffness", "", ObservationType.JOINT_GAIN_STIFFNESS, self._action_spec),
            ("joint_default_pos", "", ObservationType.JOINT_DEFAULT_POS, self._action_spec),
            ("robot_mass", "", ObservationType.SUB_BODY_MASS, sub_bodies),
        ]

        collision_groups = [
            ("feet", ["/fl_foot", "/fr_foot", "/rl_foot", "/rr_foot"]), 
            ("body", ["/body", "/fl_l1", "/fr_l1", "/rl_l1", "/rr_l1"]),
            ("lower_body", ["/fl_l2", "/fr_l2", "/rl_l2", "/rr_l2"])
        ]

        collision_between_envs = False
        env_spacing = 3.
        physics_material_spec = self._get_values_for_physics_materials(num_envs) if domain_randomization else None
        sim_params = {
            "gpu_found_lost_aggregate_pairs_capacity": 128*1024, 
            "gpu_total_aggregate_pairs_capacity": 128*1024, 
            "gpu_temp_buffer_capacity": 16777216,
            "gpu_max_rigid_patch_count": 2 * 81920,
        }

        solver_pos = torch.full((num_envs, ), 4)
        solver_vel = torch.full((num_envs, ), 0)
        super().__init__(usd_path, self._action_spec, observation_spec, backend, device, collision_between_envs, num_envs, 
                         env_spacing, 0.99, horizon, additional_data_spec=additional_data_spec, collision_groups=collision_groups, 
                         action_type=ActionType.EFFORT, headless=headless, n_intermediate_steps=4, timestep=0.005, 
                         physics_material_spec=physics_material_spec, sim_params=sim_params, camera_position=camera_pos, 
                         camera_target=camera_target, solver_pos_it_count=solver_pos, solver_vel_it_count=solver_vel) 
        self._import_helper_functions()
        if self.domain_randomization:
            self._init_domain_randomization_parameters()

        #update action space
        action_limits = (self._task.get_joint_pos_limits() - self._default_joint_angles) / 0.25
        self._mdp_info.action_space = Box(*action_limits, data_type=action_limits[0].dtype)
        
        #register custom observations
        self.observation_helper.add_obs("projected_gravity", 3, -1, 1)
        commands_upper = torch.tensor([1., 1., np.pi], device=device)
        self.observation_helper.add_obs("commands", 3, -commands_upper, commands_upper)
        self.observation_helper.add_obs("actions", self.NUM_JOINTS, self.info.action_space.low, self.info.action_space.high)

        if self.domain_randomization:
            self.add_domain_randomization_observations()

        #get normalization and noise vector
        self._normalization_obs_vec = self._get_obs_normilization_vec()
        self._noise_scale_vec = self._get_noise_scale_vec()
        self._soft_joint_pos_limits = self._get_soft_joint_pos_limit()

        #update observation space
        obs_low, obs_high = self.observation_helper.obs_limits
        joint_pos_indices = self.observation_helper.obs_idx_map["joint_pos"]
        obs_low[joint_pos_indices] -= self._default_joint_angles
        obs_high[joint_pos_indices] -= self._default_joint_angles
        new_obs_low = obs_low * self._normalization_obs_vec - self._noise_scale_vec
        new_obs_high = obs_high * self._normalization_obs_vec + self._noise_scale_vec
        self._mdp_info.observation_space = Box(new_obs_low, new_obs_high, data_type=new_obs_high.dtype)

        self._commands = torch.zeros(num_envs, 4, dtype=torch.float, device=device)
        self._actions = torch.zeros((num_envs, self.NUM_JOINTS), device=device)
        self._feet_air_time = torch.zeros((num_envs, 4), device=device)
        self._last_actions =  torch.zeros((num_envs, self.NUM_JOINTS), device=device)
        self._last_joint_vel = torch.zeros((num_envs, self.NUM_JOINTS), device=device)
        self._last_contacts = torch.zeros((num_envs, 4), device=device, dtype=torch.bool)
        self._episode_length = torch.zeros((num_envs, ), dtype=int, device=device)
        self._forward_vec = torch.tensor([1., 0., 0.], device=device).repeat((num_envs, 1))
        self._gravity = torch.tensor([0., 0., -1.], device=self._device).repeat((self.number, 1))
        self._effort_limit = self._task.get_joint_max_efforts()

        #domain randomization
        self._np_rng = np.random.default_rng()
        self._current_mixed = False
        self._current_nr_delay_steps = 0
        self._action_history = torch.zeros((self.MAX_NR_DELAY_STEPS + 1, self.number, self.NUM_JOINTS), device=self._device)
    
    def _import_helper_functions(self):
        from isaacsim.core.utils.torch.rotations import quat_apply, quat_rotate_inverse
        from isaacsim.core.utils.torch.maths import torch_rand_float
        self.quat_apply = quat_apply
        self.quat_rotate_inverse = quat_rotate_inverse
        self.torch_rand_float = torch_rand_float

    def is_absorbing(self, obs):
        forces = self._get_net_collision_forces("body", dt=self._timestep)
        fallen = torch.any(torch.norm(forces, dim=-1) > 0., dim=-1)
        return fallen
    
    def setup(self, env_indices, obs):
        #new
        self._feet_air_time[env_indices] = 0.
        self._episode_length[env_indices] = 0

        if self.domain_randomization:
            joint_pos = self._seen_joint_nominal_pos[env_indices]
        else:
            r_factors = self.torch_rand_float(0.5, 1.5, (len(env_indices), self.NUM_JOINTS), device=self._device)
            joint_pos = self._default_joint_angles * r_factors
        joint_vel = torch.zeros((len(env_indices), len(self._action_spec)), device=self._device)

        self._write_data("joint_pos", joint_pos, env_indices)
        self._write_data("joint_vel", joint_vel, env_indices)

        body_vel = self.torch_rand_float(-0.5, 0.5, (len(env_indices), 6), device=self._device)
        self._write_data("body_vel", body_vel, env_indices)

        self._setup_joint_pos = joint_pos
        self._setup_joint_vel = joint_vel
        self._setup_env_indices = env_indices

        #update last_joint_vel
        self._last_joint_vel[env_indices] = joint_vel

        self._resample_commands(env_indices)

        zero = torch.zeros(self._n_envs, device=self._device)
        self._extra_info_rewards = self._extra_info_rewards = {
            "r_tracking_lin_vel": zero, "r_tracking_ang_vel": zero, "r_lin_vel_z": zero,
            "r_ang_vel_xy": zero, "r_torques": zero, "r_joint_acc": zero, "r_feet_air_time": zero,
            "r_collision": zero, "r_action_rate": zero, "r_joint_pos_limits": zero
        }

        self._action_history[:, env_indices, :] = 0
    
    def _step_finalize(self, env_indices):
        self._episode_length += 1

        #resample commands
        do_resample = self.torch_rand_float(0., 1., (len(env_indices), 1), device=self._device).squeeze(-1) < (1./500.)
        do_resample *= self._episode_length[env_indices] > 50
        env_ids = env_indices[do_resample]
        self._resample_commands(env_ids)

        #calculate yaw command
        base_quat = self._read_data("body_rot")
        forward = self.quat_apply(base_quat, self._forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self._commands[:, 2] = torch.clip(0.5*self.wrap_to_pi(self._commands[:, 3] - heading), -1., 1.)

        #domain randomization: push Robot
        if self.domain_randomization:
            do_push = self.torch_rand_float(0., 1., (len(env_indices), 1), device=self._device).squeeze(-1) < (1./750.)
            do_push_ids = env_indices[do_push]
            do_push_ids = do_push_ids[self._episode_length[do_push_ids] > 50]
            
            self._push_robots(do_push_ids)

            if self._np_rng.uniform() < 0.002:
                self._current_mixed = self._np_rng.uniform() < self.MIXED_CHANCE
                self._current_nr_delay_steps = 0

            if self._np_rng.uniform() < 0.0004:
                self.sample_unseen_noise_factors(torch.arange(0, self.number, 1, device=self._device))
                self.sample_seen_parameters(torch.arange(0, self.number, 1, device=self._device))
    
    def _resample_commands(self, env_ids):
        self._commands[env_ids, 0] = self.torch_rand_float(-1., 1., (len(env_ids), 1), device=self._device).squeeze(1)
        self._commands[env_ids, 1] = self.torch_rand_float(-1., 1., (len(env_ids), 1), device=self._device).squeeze(1)
        self._commands[env_ids, 3] = self.torch_rand_float(-3.14, 3.14, (len(env_ids), 1), device=self._device).squeeze(1)

        # set small commands to zero
        self._commands[env_ids, :2] *= (torch.norm(self._commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    @staticmethod
    def wrap_to_pi(angles):
        angles %= 2*np.pi
        angles -= 2*np.pi * (angles > np.pi)
        return angles

    def _get_obs_normilization_vec(self):
        v = torch.ones((self.observation_helper.obs_length), device=self._device)

        lin_vel = self.observation_helper.obs_idx_map["base_lin_vel"]
        ang_vel = self.observation_helper.obs_idx_map["base_ang_vel"]
        joint_positions = self.observation_helper.obs_idx_map["joint_pos"]
        joint_velocities = self.observation_helper.obs_idx_map["joint_vel"]
        gravity = self.observation_helper.obs_idx_map["projected_gravity"]
        commands = self.observation_helper.obs_idx_map["commands"]
        actions = self.observation_helper.obs_idx_map["actions"]
        pos = self.observation_helper.obs_idx_map["base_pos"]
        
        v[lin_vel] = 2.0
        v[ang_vel] = 0.25
        v[joint_positions] = 1.00
        v[joint_velocities] = 0.05
        v[gravity] = 1.
        v[commands[0:2]] = 2.0
        v[commands[2]] = 0.25
        v[actions] = 1.
        v[pos] = 1 / 0.4

        if self.domain_randomization:
            joint_nominal_pos_ids = self.observation_helper.obs_idx_map["joint_nominal_position"]
            torque_limit_ids = self.observation_helper.obs_idx_map["torque_limit"]
            joint_max_velocity_ids = self.observation_helper.obs_idx_map["joint_max_velocity"]
            joint_damping_ids = self.observation_helper.obs_idx_map["joint_damping"]
            joint_stiffness_ids = self.observation_helper.obs_idx_map["joint_stiffness"]
            joint_armature_ids = self.observation_helper.obs_idx_map["joint_armature"]
            joint_frictionloss_ids = self.observation_helper.obs_idx_map["joint_frictionloss"]
            p_gain_ids = self.observation_helper.obs_idx_map["p_gain"]
            d_gain_ids = self.observation_helper.obs_idx_map["d_gain"]
            action_scaling_factor_ids = self.observation_helper.obs_idx_map["action_scaling_factor"]
            mass_ids = self.observation_helper.obs_idx_map["mass"]

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

    def _get_noise_scale_vec(self):
        v = torch.zeros((self.observation_helper.obs_length), device=self._device)

        lin_vel = self.observation_helper.obs_idx_map["base_lin_vel"]
        ang_vel = self.observation_helper.obs_idx_map["base_ang_vel"]
        joint_positions = self.observation_helper.obs_idx_map["joint_pos"]
        joint_velocities = self.observation_helper.obs_idx_map["joint_vel"]
        gravity = self.observation_helper.obs_idx_map["projected_gravity"]
        commands = self.observation_helper.obs_idx_map["commands"]
        actions = self.observation_helper.obs_idx_map["actions"]
        
        v[lin_vel] = 0.1 * 2.0
        v[ang_vel] = 0.2 * 0.25
        v[joint_positions] = 0.01 * 1.00
        v[joint_velocities] = 1.5 * 0.05
        v[gravity] = 0.05
        v[commands[:3]] = 0
        v[actions] = 0

        return v
    
    def _get_soft_joint_pos_limit(self):
        soft_joint_pos_limits = torch.zeros(self.NUM_JOINTS, 2, device=self._device, requires_grad=False)
        pos_limit = self._task.get_joint_pos_limits()
        low = pos_limit[0]
        high = pos_limit[1]
        
        middle = (low + high) / 2
        r = high - low
        soft_joint_pos_limits[:, 0] = middle - 0.5 * r * 0.9
        soft_joint_pos_limits[:, 1] = middle + 0.5 * r * 0.9
        return soft_joint_pos_limits

    # observations -------------------------------------------------------------------------------------
    def _create_observation(self, obs):
        #update observation with values set in setup
        if self._setup_env_indices is not None:
            joint_pos_indices = self.observation_helper.obs_idx_map["joint_pos"]
            obs[self._setup_env_indices.unsqueeze(1), joint_pos_indices] = self._setup_joint_pos

            joint_vel_indices = self.observation_helper.obs_idx_map["joint_vel"]
            obs[self._setup_env_indices.unsqueeze(1), joint_vel_indices] = self._setup_joint_vel

            self._setup_env_indices = None

        #set missing observations
        rot = self._read_data("body_rot")
        gravity_indices = self.observation_helper.obs_idx_map["projected_gravity"]
        obs[:, gravity_indices] = self.quat_rotate_inverse(rot, self._gravity)

        command_indices = self.observation_helper.obs_idx_map["commands"]
        obs[:, command_indices] = self._commands[:, :3]

        action_indices = self.observation_helper.obs_idx_map["actions"]
        obs[:, action_indices] = self._actions

        lin_vel_indices = self.observation_helper.obs_idx_map["base_lin_vel"] 
        lin_vel = self.observation_helper.get_from_obs(obs, "base_lin_vel")
        obs[:, lin_vel_indices] = self.quat_rotate_inverse(rot, lin_vel)

        ang_vel_indices = self.observation_helper.obs_idx_map["base_ang_vel"]
        ang_vel = self.observation_helper.get_from_obs(obs, "base_ang_vel")
        obs[:, ang_vel_indices] = self.quat_rotate_inverse(rot, ang_vel)

        base_pos_indices = self.observation_helper.obs_idx_map["base_pos"]
        obs[:, base_pos_indices[:2]] = 0

        return obs

    def _modify_observation(self, obs):
        obs = self._add_seen_parameters(obs)

        joint_pos_indices = self.observation_helper.obs_idx_map["joint_pos"]
        obs[:, joint_pos_indices] -= self._default_joint_angles

        command_indices = self.observation_helper.obs_idx_map["commands"]
        obs[:, command_indices] = self._commands[:, :3]

        obs *= self._normalization_obs_vec
        obs += (2 * torch.rand_like(obs) - 1) * self._noise_scale_vec

        obs = torch.clamp(obs, max=100., min=-100.)

        return obs

    def _create_info_dictionary(self, obs):
        return self._extra_info_rewards
    
    # control ---------------------------------------------------------------------------------------
    def _preprocess_action(self, action):
        action = torch.clip(action, min=-100., max=100.)
        if self.domain_randomization:
            action = self.delay_action(action)
        self._actions[:] = action[:]
        return action
    
    def _compute_action(self, action):
        joint_vels = self._read_data("joint_vel")
        joint_positions = self._read_data("joint_pos")
        torque = self._compute_torque(action, joint_vels, joint_positions)
        return torque
    
    def _compute_torque(self, action, joint_vels, joint_pos):
        action_scaled = action * self._seen_scaling_factor
        target_joint_pos = self._seen_joint_nominal_pos + action_scaled

        self._torques = self._unseen_p_gain * (target_joint_pos - joint_pos + self._joint_position_offset) \
            - self._unseen_d_gain * joint_vels
        self._torques *= self._nf_motor_strength
        self._torques = torch.clip(self._torques, -self._seen_torque_limit, self._seen_torque_limit)

        return self._torques
    
    #Taken from https://proceedings.mlr.press/v164/rudin22a.html
    #Taken from https://github.com/leggedrobotics/legged_gym/blob/17847702f90d8227cd31cce9c920aa53a739a09a/legged_gym/envs/base/legged_robot.py#L815C3-L816C12
    def reward(self, obs, action, next_obs, absorbing):
        base_lin_vel = self.observation_helper.get_from_obs(next_obs, "base_lin_vel")
        base_lin_vel_xy = base_lin_vel[:, 0:2]
        base_lin_vel_z = base_lin_vel[:, 2]
        base_ang_vel = self.observation_helper.get_from_obs(next_obs, "base_ang_vel")
        base_ang_vel_xy = base_ang_vel[:, 0:2]
        base_ang_vel_z = base_ang_vel[:, 2]

        joint_vel = self.observation_helper.get_from_obs(next_obs, "joint_vel")
        joint_pos = self.observation_helper.get_from_obs(next_obs, "joint_pos")

        base_pos = self.observation_helper.get_from_obs(next_obs, "base_pos")
        base_pos_z = base_pos[:, 2]

        #---------------------------------------------------------------------------

        r_tracking_lin_vel = self._reward_tracking_lin_vel(base_lin_vel_xy) * 1.0 * self.dt
        r_tracking_ang_vel = self._reward_tracking_ang_vel(base_ang_vel_z) * 0.5 * self.dt
        r_lin_vel_z = self._reward_lin_vel_z(base_lin_vel_z) * -2.0 * self.dt
        r_ang_vel_xy = self._reward_ang_vel_xy(base_ang_vel_xy) * -0.05 * self.dt
        r_torques = self._reward_torques(self._torques) * -0.0001 * self.dt
        r_joint_acc = self._reward_joint_acc(joint_vel) * -2.5e-7 * self.dt
        r_feet_air_time = self._reward_feet_air_time() * 1.0 * self.dt
        r_collision = self._reward_collision() * -1. * self.dt
        r_action_rate = self._reward_action_rate(action) * -0.01 * self.dt
        r_joint_pos_limits = self._reward_joint_pos_limits(joint_pos) * -10.0 * self.dt
        r_height = self._reward_height(base_pos_z) * -4. * self.dt

        self._extra_info_rewards = {
            "tracking_lin_vel": r_tracking_lin_vel, "tracking_ang_vel": r_tracking_ang_vel,
            "lin_vel_z": r_lin_vel_z, "ang_vel_xy": r_ang_vel_xy, 
            "torques": r_torques, "joint_acc": r_joint_acc, 
            "feet_air_time": r_feet_air_time, "collision": r_collision, 
            "action_rate": r_action_rate, "joint_pos_limits": r_joint_pos_limits
        }

        reward = r_tracking_lin_vel + r_tracking_ang_vel + r_lin_vel_z + r_ang_vel_xy + r_torques + r_joint_acc + r_feet_air_time \
                + r_collision + r_action_rate + r_joint_pos_limits + r_height
        
        reward = torch.clamp(reward, min=0.)

        self._last_actions = action.clone().detach()
        self._last_joint_vel = joint_vel.clone().detach()
        
        return reward
    
    def _reward_lin_vel_z(self, lin_vel_z):
        # Penalize z axis base linear velocity
        return torch.square(lin_vel_z)
    
    def _reward_ang_vel_xy(self, base_ang_vel_xy):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(base_ang_vel_xy), dim=1)
    
    def _reward_torques(self, torques):
        # Penalize torques
        return torch.sum(torch.square(torques), dim=1)
    
    def _reward_joint_acc(self, joint_vel):
        # Penalize joint accelerations
        return torch.sum(torch.square((self._last_joint_vel - joint_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self, actions):
        # Penalize changes in actions
        return torch.sum(torch.square(self._last_actions - actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        forces = self._get_net_collision_forces("lower_body", dt=self._timestep)
        contact = torch.norm(forces, dim=-1) > 0.1
        return torch.sum(contact, dim=1)
    
    def _reward_joint_pos_limits(self, joint_pos):
        # Penalize joint positions too close to the limit
        out_of_limits = -(joint_pos - self._soft_joint_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (joint_pos - self._soft_joint_pos_limits[:, 1]).clip(min=0.) # upper limit
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel(self, lin_vel_xy):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - lin_vel_xy), dim=1)
        return torch.exp(-lin_vel_error/0.25)
    
    def _reward_tracking_ang_vel(self, ang_vel_z):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self._commands[:, 2] - ang_vel_z)
        return torch.exp(-ang_vel_error/0.25)

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self._get_net_collision_forces("feet", dt=self._timestep)[:, :, 2] > 1.
        contact_filt = torch.logical_or(contact, self._last_contacts)
        self._last_contacts = contact
        first_contact = (self._feet_air_time > 0.) * contact_filt
        self._feet_air_time += self.dt
        rew_airTime = torch.sum((self._feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self._commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self._feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_height(self, base_z):
        #nominal_base_z = 0.316
        nominal_base_z = 0.31
        return torch.square(base_z - nominal_base_z)
    
    # Domain Randomization ----------------------------------------------------------------------------------
    def _get_values_for_physics_materials(self, num_envs):
        friction_range = [0.5, 1.25]
        num_buckets = 64
        bucket_ids = torch.randint(0, num_buckets, (num_envs, ))
        friction_buckets = (friction_range[1] - friction_range[0]) * torch.rand((num_buckets, ), device='cpu') + friction_range[0]
        
        names = [f"custom_material_{i}" for i in bucket_ids.tolist()]
        dynamic_friction = [0.5] * num_envs
        static_friction = friction_buckets[bucket_ids].tolist()
        restitution = [0.0] * num_envs
        
        return list(zip(names, dynamic_friction, static_friction, restitution))
    
    def _push_robots(self, env_indices):
        max_vel= 1.
        vels = self.torch_rand_float(-max_vel, max_vel, (env_indices.shape[0], 2), device=self._device)
        extended_vels = self._read_data("body_vel", env_indices)
        extended_vels[:, :2] = vels
        self._write_data("body_vel", extended_vels, env_indices)
    
    def delay_action(self, action):
        if self._current_mixed:
            self._current_nr_delay_steps = self._np_rng.integers(self.MAX_NR_DELAY_STEPS+1)

        self._action_history = torch.roll(self._action_history, -1, dims=0)
        self._action_history[-1] = action

        chosen_action = self._action_history[-1 - self._current_nr_delay_steps]

        return chosen_action
    
    def _init_domain_randomization_parameters(self):
        #init some seen parameters
        self._seen_joint_damping = self._read_data("joint_damping")
        self._seen_joint_stiffness = self._read_data("joint_stiffness")
        self._seen_joint_armature = self._read_data("joint_armature")
        self._seen_joint_frictionloss = self._read_data("joint_frictionloss")

        self._seen_mass = self._read_data("robot_mass")
        self._seen_summed_mass = torch.sum(self._seen_mass, dim=1)
        self._seen_torque_limit = self._read_data("torque_limit")
        self._seen_joint_nominal_pos = self._default_joint_angles.repeat((self.number, 1))
        self._seen_joint_max_vel = self._default_joint_max_vel.repeat((self.number, 1))
        self._seen_foot_scaling = torch.ones((self.number, 4), device=self._device)
        self.seen_trunk_com = self._read_data("trunk_com")

        self._write_data("max_joint_vel", self._seen_joint_max_vel, reapply_after_reset=True)

        self._seen_p_gain = torch.full((self.number, self.NUM_JOINTS), 20., device=self._device)
        self._seen_d_gain = torch.full((self.number, self.NUM_JOINTS), 0.5, device=self._device)
        self._seen_scaling_factor = torch.full((self.number, self.NUM_JOINTS), 0.25, device=self._device)

        self._unseen_p_gain = torch.full((self.number, self.NUM_JOINTS), 20., device=self._device)
        self._unseen_d_gain = torch.full((self.number, self.NUM_JOINTS), 0.5, device=self._device)

        self._default_trunk_mass = self._read_data("trunk_mass")[0].clone().detach()
        self._default_trunk_inertia = self._read_data("trunk_inertia")[0].clone().detach()
        self._default_trunk_com = self._read_data("trunk_com")[0].clone().detach()
        self._default_torque_limit = self._seen_torque_limit[0].clone().detach()
        self._default_joint_nominal_pos = self._seen_joint_nominal_pos[0].clone().detach()
        #self._default_joint_max_vel = self._seen_joint_max_vel[0]
        self._default_joint_range = self._read_data("joint_range")[0].clone().detach()
        self._default_joint_damping = self._seen_joint_damping[0].clone().detach()
        self._default_joint_stiffness = self._seen_joint_stiffness[0].clone().detach()
        self._default_joint_armature = self._seen_joint_armature[0].clone().detach()
        self._default_joint_frictionloss = self._seen_joint_frictionloss[0].clone().detach()

        self._nf_trunk_mass = torch.ones((self.number, 1), device=self._device)
        self._nf_trunk_com = torch.ones((self.number, 1), device=self._device)
        self._nf_foot_size = torch.ones((self.number, 1), device=self._device)
        self._nf_joint_damping = torch.ones((self.number, 1), device=self._device)
        self._nf_joint_stiffness = torch.ones((self.number, 1), device=self._device)
        self._nf_joint_armature = torch.ones((self.number, 1), device=self._device)
        self._nf_joint_friction = torch.ones((self.number, 1), device=self._device)

        self._nf_p_gain = torch.ones((self.number, 1), device=self._device)
        self._nf_d_gain = torch.ones((self.number, 1), device=self._device)
        self._nf_motor_strength = torch.ones((self.number, 1), device=self._device)
        self._joint_position_offset = torch.zeros((self.number, self.NUM_JOINTS), device=self._device)

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
            position_offset=0.05
        ):
        n_envs = env_indices.shape[0]

        self._nf_trunk_mass[env_indices] = self.torch_rand_float(1 - trunk_mass_factor, 1 + trunk_mass_factor, (n_envs, 1), self._device)
        self._nf_trunk_com[env_indices] = self.torch_rand_float(1 - trunk_com_factor, 1 + trunk_com_factor, (n_envs, 1), self._device)
        self._nf_foot_size[env_indices] = self.torch_rand_float(1 - foot_size_factor, 1 + foot_size_factor, (n_envs, 1), self._device)
        self._nf_joint_damping[env_indices] = self.torch_rand_float(1 - joint_damping_factor, 1 + joint_damping_factor, (n_envs, 1), self._device)
        self._nf_joint_stiffness[env_indices] = self.torch_rand_float(1 - joint_stiffness_factor, 1 + joint_stiffness_factor, (n_envs, 1), self._device)
        self._nf_joint_armature[env_indices] = self.torch_rand_float(1 - joint_armature_factor, 1 + joint_armature_factor, (n_envs, 1), self._device)
        self._nf_joint_friction[env_indices] = self.torch_rand_float(1 - joint_friction_factor, 1 + joint_friction_factor, (n_envs, 1), self._device)

        #control function
        self._nf_p_gain[env_indices] = self.torch_rand_float(1 - p_gain_factor, 1 + p_gain_factor, (n_envs, 1), self._device)
        self._nf_d_gain[env_indices] = self.torch_rand_float(1 - d_gain_factor, 1 + d_gain_factor, (n_envs, 1), self._device)
        self._nf_motor_strength[env_indices] = self.torch_rand_float(1 - motor_strength_factor, 1 + motor_strength_factor, (n_envs, 1), self._device)
        self._joint_position_offset[env_indices] = self.torch_rand_float(-position_offset, position_offset, (n_envs, self.NUM_JOINTS), self._device)
    
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
            add_scaling_factor_min=-0.03, add_scaling_factor_max=0.03,  
        ):
        n_envs = env_indices.shape[0]

        #trunk mass
        trunk_mass = self._default_trunk_mass \
            + self.torch_rand_float(add_trunk_mass_min, add_trunk_mass_max, (n_envs, 1), self._device)
        actual_trunk_mass = trunk_mass * self._nf_trunk_mass[env_indices]
        self._write_data("trunk_mass", actual_trunk_mass, env_indices, True)
        actual_trunk_inertia = self._default_trunk_inertia + (actual_trunk_mass / self._default_trunk_mass)
        self._write_data("trunk_inertia", actual_trunk_inertia.unsqueeze(1), env_indices, True)
        self._seen_mass[env_indices, 0] = trunk_mass.squeeze(1)
        self._seen_summed_mass = torch.sum(self._seen_mass, dim=1)

        #trunk com
        actual_trunk_com = self._default_trunk_com \
            + self.torch_rand_float(add_com_displacement_min, add_com_displacement_max, (n_envs, 1), self._device)
        self.seen_trunk_com[env_indices] = actual_trunk_com.unsqueeze(1)
        actual_trunk_com *= self._nf_trunk_com[env_indices]
        self._write_data("trunk_com", actual_trunk_com.unsqueeze(1), env_indices, True)

        #foot scaling
        self._seen_foot_scaling  = self.torch_rand_float(foot_scaling_min, foot_scaling_max, (n_envs, 4), self._device)
        actual_foot_scaling = self._seen_foot_scaling * self._nf_foot_size[env_indices]
        for i, name in enumerate(["FL_foot_scale", "FR_foot_scale", "RL_foot_scale", "RR_foot_scale"]):
            self._write_data(name, actual_foot_scaling[env_indices, i].unsqueeze(1).repeat(1, 3), env_indices, True)
        
        #joint nominal position
        self._seen_joint_nominal_pos[env_indices] = self._default_joint_nominal_pos \
            + self.torch_rand_float(add_joint_nominal_position_min, add_joint_nominal_position_max, (n_envs, self.NUM_JOINTS), self._device)
        #self._write_data("joint_default_pos", self._seen_joint_nominal_pos[env_indices], env_indices)

        #joint torque limit
        self._seen_torque_limit[env_indices] = self._default_torque_limit \
            * (1 + self.torch_rand_float(-torque_limit_factor, torque_limit_factor, (n_envs, self.NUM_JOINTS), self._device))
        self._write_data("torque_limit", self._seen_torque_limit[env_indices], env_indices, True)

        #joint max velocity
        self._seen_joint_max_vel[env_indices] = self._default_joint_max_vel \
            * (1 + self.torch_rand_float(-joint_velocity_factor, joint_velocity_factor, (n_envs, self.NUM_JOINTS), self._device))
        self._write_data("max_joint_vel", self._seen_joint_max_vel[env_indices], env_indices, True)

        #joint damping, stiffness, armature, frictionloss
        stay_at_default_mask = self.torch_rand_float(0, 1, (n_envs, 1), self._device) < stay_at_default_percentage
        stay_at_default_mask = stay_at_default_mask.squeeze()
        stay_at_default_idx = env_indices[stay_at_default_mask]
        self._seen_joint_damping[stay_at_default_idx] = self._default_joint_damping
        self._seen_joint_stiffness[stay_at_default_idx] = self._default_joint_stiffness
        self._seen_joint_armature[stay_at_default_idx] = self._default_joint_armature
        self._seen_joint_frictionloss[stay_at_default_idx] = self._default_joint_frictionloss

        not_stay_at_default_mask = torch.logical_not(stay_at_default_mask)
        not_stay_at_default_idx = env_indices[not_stay_at_default_mask]
        num_envs_not_default = not_stay_at_default_idx.shape[0]
        self._seen_joint_damping[not_stay_at_default_idx] = self.torch_rand_float(joint_damping_min, joint_damping_max, (num_envs_not_default, self.NUM_JOINTS), self._device)
        self._seen_joint_stiffness[not_stay_at_default_idx] = self.torch_rand_float(joint_stiffness_min, joint_stiffness_max, (num_envs_not_default, self.NUM_JOINTS), self._device)
        self._seen_joint_armature[not_stay_at_default_idx] = self.torch_rand_float(joint_armature_min, joint_armature_max, (num_envs_not_default, self.NUM_JOINTS), self._device)
        self._seen_joint_frictionloss[not_stay_at_default_idx] = self.torch_rand_float(joint_friction_loss_min, joint_friction_loss_max, (num_envs_not_default, self.NUM_JOINTS), self._device)

        self._write_data("joint_damping", self._seen_joint_damping[env_indices] * self._nf_joint_damping[env_indices], env_indices, True) #chceck if damping is difference in scale
        self._write_data("joint_stiffness", self._seen_joint_stiffness[env_indices] * self._nf_joint_stiffness[env_indices], env_indices, True)
        self._write_data("joint_armature", self._seen_joint_armature[env_indices] * self._nf_joint_armature[env_indices], env_indices, True)
        self._write_data("joint_frictionloss", self._seen_joint_frictionloss[env_indices] * self._nf_joint_friction[env_indices], env_indices, True)

        #used for control function
        self._seen_p_gain[env_indices] = 20 + self.torch_rand_float(add_p_gain_min, add_p_gain_max, (n_envs, self.NUM_JOINTS), self._device)
        self._seen_d_gain[env_indices] = 0.5 + self.torch_rand_float(add_d_gain_min, add_d_gain_max, (n_envs, self.NUM_JOINTS), self._device)
        self._seen_scaling_factor[env_indices] = 0.25 + self.torch_rand_float(add_scaling_factor_min, add_scaling_factor_max, (n_envs, self.NUM_JOINTS), self._device)

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
            add_scaling_factor_min=-0.03, add_scaling_factor_max=0.03,  
        ):
        #joints
        self.observation_helper.add_obs(
            name="joint_nominal_position", 
            length=self.NUM_JOINTS, 
            min_value=(self._default_joint_angles + add_joint_nominal_position_min) / 4.6, 
            max_value=(self._default_joint_angles + add_joint_nominal_position_max) / 4.6
        )
        self.observation_helper.add_obs(
            name="torque_limit", 
            length=self.NUM_JOINTS, 
            min_value=(self._default_torque_limit * (1 - torque_limit_factor)) / (1000.0 / 2) - 1.0, 
            max_value=(self._default_torque_limit * (1. + torque_limit_factor)) / (1000.0 / 2) - 1.0
        )
        self.observation_helper.add_obs(
            name="joint_max_velocity", 
            length=self.NUM_JOINTS, 
            min_value=(self._default_joint_max_vel * (1 - joint_velocity_factor)) / (35.0 / 2) - 1.0,
            max_value=(self._default_joint_max_vel * (1 + joint_velocity_factor)) / (35.0 / 2) - 1.0
        )
        self.observation_helper.add_obs(
            name="joint_damping",
            length=self.NUM_JOINTS,
            min_value=joint_damping_min / (10.0 / 2) - 1.0,
            max_value=joint_damping_max / (10.0 / 2) - 1.0
        )
        self.observation_helper.add_obs(
            name="joint_stiffness",
            length=self.NUM_JOINTS,
            min_value=joint_stiffness_min / (30.0 / 2) - 1.0,
            max_value=joint_stiffness_max / (30.0 / 2) - 1.0
        )
        self.observation_helper.add_obs(
            name="joint_armature",
            length=self.NUM_JOINTS,
            min_value=joint_armature_min / (0.2 / 2) - 1.0,
            max_value=joint_armature_max / (0.2 / 2) - 1.0
        )
        self.observation_helper.add_obs(
            name="joint_frictionloss",
            length=self.NUM_JOINTS,
            min_value=joint_friction_loss_min / (1.2 / 2) - 1.0,
            max_value=joint_friction_loss_max / (1.2 / 2) - 1.0
        )
        self.observation_helper.add_obs(
            name="p_gain",
            length=self.NUM_JOINTS,
            min_value=20 + add_p_gain_min / (100.0 / 2) - 1.0,
            max_value=20 + add_p_gain_max / (100.0 / 2) - 1.0
        )
        self.observation_helper.add_obs(
            name="d_gain",
            length=self.NUM_JOINTS,
            min_value=0.5 + add_d_gain_min / (2.0 / 2) - 1.0,
            max_value=0.5 + add_d_gain_max / (2.0 / 2) - 1.0
        )
        self.observation_helper.add_obs(
            name="action_scaling_factor",
            length=self.NUM_JOINTS,
            min_value=0.25 + add_scaling_factor_min / (0.8 / 2) - 1.0,
            max_value=0.25 + add_scaling_factor_max / (0.8 / 2) - 1.0
        )

        #mass, com foot scaling
        self.observation_helper.add_obs(
            name="mass",
            length=1,
            min_value=-torch.inf,
            max_value=torch.inf
        )

    def _add_seen_parameters(self, obs):
        joint_nominal_pos_ids = self.observation_helper.obs_idx_map["joint_nominal_position"]
        obs[:, joint_nominal_pos_ids] = self._seen_joint_nominal_pos

        torque_limit_ids = self.observation_helper.obs_idx_map["torque_limit"]
        obs[:, torque_limit_ids] = self._seen_torque_limit

        joint_max_velocity_ids = self.observation_helper.obs_idx_map["joint_max_velocity"]
        obs[:, joint_max_velocity_ids] = self._seen_joint_max_vel
        
        joint_damping_ids = self.observation_helper.obs_idx_map["joint_damping"]
        obs[:, joint_damping_ids] = self._seen_joint_damping

        joint_stiffness_ids = self.observation_helper.obs_idx_map["joint_stiffness"]
        obs[:, joint_stiffness_ids] = self._seen_joint_stiffness

        joint_armature_ids = self.observation_helper.obs_idx_map["joint_armature"]
        obs[:, joint_armature_ids] = self._seen_joint_armature

        joint_frictionloss_ids = self.observation_helper.obs_idx_map["joint_frictionloss"]
        obs[:, joint_frictionloss_ids] = self._seen_joint_frictionloss

        p_gain_ids = self.observation_helper.obs_idx_map["p_gain"]
        obs[:, p_gain_ids] = self._seen_p_gain

        d_gain_ids = self.observation_helper.obs_idx_map["d_gain"]
        obs[:, d_gain_ids] = self._seen_d_gain

        action_scaling_factor_ids = self.observation_helper.obs_idx_map["action_scaling_factor"]
        obs[:, action_scaling_factor_ids] = self._seen_scaling_factor
        
        mass_ids = self.observation_helper.obs_idx_map["mass"]
        obs[:, mass_ids] = self._seen_summed_mass.unsqueeze(1)
        
        return obs