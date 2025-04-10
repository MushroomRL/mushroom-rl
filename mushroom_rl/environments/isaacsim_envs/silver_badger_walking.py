import numpy as np
import torch
from pathlib import Path

from mushroom_rl.environments import IsaacSim
from mushroom_rl.utils.isaac_sim import ObservationType, ActionType
from mushroom_rl.environments.isaacsim_envs.honey_badger_walking import HoneyBadgerWalking
from mushroom_rl.rl_utils.spaces import Box

class SilverBadgerWalking(HoneyBadgerWalking):
    """
    A learning environment for training the Silver Badger quadroped to walk. 
    Silver Badger is a Robot from MAB Robotics: https://www.mabrobotics.pl/
    """
    def __init__(self, num_envs, horizon, headless, domain_randomization=True, camera_pos=(105, 0, 4), camera_target=(95, 0, 0)):
        usd_path = str(Path(__file__).resolve().parent / "robots_usds/silver_badger/silver_badger.usd")
        self.NUM_JOINTS = 13

        backend="torch"
        device="cuda:0"
        self.domain_randomization = domain_randomization

        self._action_spec = [
            "fl_j0", "fl_j1", "fl_j2",
            "fr_j0", "fr_j1", "fr_j2",   
            "rl_j0", "rl_j1", "rl_j2", 
            "rr_j0", "rr_j1", "rr_j2", 
            "sp_j0"
        ]
        self._default_joint_angles = torch.tensor([
            0.1, -0.8, 1.5,
            -0.1, 0.8, -1.5,
            0.1, -1., 1.5,
            -0.1, 1., -1.5, 
            0
        ], device=device)
        self._default_joint_max_vel = torch.tensor([
            25., 25., 25.,
            25., 25., 25.,
            25., 25., 25.,
            25., 25., 25.,
            25.
        ],device=device)
        
        observation_spec = [
            ("base_lin_vel", "", ObservationType.BODY_LIN_VEL, None),
            ("base_ang_vel", "", ObservationType.BODY_ANG_VEL, None),
            ("joint_pos", "", ObservationType.JOINT_POS, self._action_spec),
            ("joint_vel", "", ObservationType.JOINT_VEL, self._action_spec),
            ("base_pos", "", ObservationType.BODY_POS, None),
        ]
        sub_bodies = [
            "body", "rear", 
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
            ("body", ["/body", "/rear", "/fl_l1", "/fr_l1", "/rl_l1", "/rr_l1"]),
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
        IsaacSim.__init__(self, usd_path, self._action_spec, observation_spec, backend, device, collision_between_envs, num_envs, 
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