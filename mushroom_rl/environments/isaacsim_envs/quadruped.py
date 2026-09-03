import math
import torch

from mushroom_rl.utils.isaac_sim.torch_maths import torch_rand_float, quat_apply, quat_mul, quat_rotate_inverse

from mushroom_rl.core.spaces import Box
from mushroom_rl.environments.isaacsim_env import IsaacSim
from mushroom_rl.environments.isaacsim_envs.quadruped_randomizer import QuadrupedRandomizationParams, \
    QuadrupedRandomizer
from mushroom_rl.utils import TorchUtils
from mushroom_rl.utils.isaac_sim import ActuationType, ObservationType


class QuadrupedIsaac(IsaacSim):
    """
    Base class for quadruped walking tasks, resembling the environment implemented by Rudin et al. for
    "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning".

    Implements the command-tracking reward terms, the joint-position-relative action space, the randomized PD
    control law and the domain randomization. A concrete quadruped supplies its USD asset, controlled joints,
    default pose, body names and collision groups, and implements ``is_absorbing`` and the collision-dependent
    reward terms.

    """
    def __init__(self, usd_path, action_spec, default_joint_angles, trunk_body, foot_bodies, sub_bodies,
                 observation_spec, additional_data_spec, collision_groups, num_envs, horizon,
                 domain_randomization, camera_position, camera_target,
                 default_joint_max_vel=None,
                 nominal_p_gain=20., nominal_d_gain=0.5, nominal_scaling_factor=0.25, reward_weights=None,
                 randomization_params=None, observed_randomization=(), max_command_ranges=None,
                 reward_params=None, clamp_reward=True,
                 command_ranges=None, tracking_stds=None, command_dead_zone=0.2,
                 command_resampling_time_range=None, heading_control_stiffness=0.5, rel_heading_envs=1.,
                 rel_standing_envs=0., frac_rotating_envs=0., frac_low_speed_envs=0., low_speed_threshold=0.5):
        """
        Constructor.

        Args:
            usd_path (str): Path to the usd file of the robot.
            action_spec (list): The names of the joints the agent controls.
            default_joint_angles (torch.tensor): The nominal joint configuration the actions are expressed
                relative to.
            trunk_body (str): The name of the trunk body, whose inertial properties are randomized.
            foot_bodies (list): The prim paths of the feet.
            sub_bodies (list): The names of every body of the robot, the trunk first.
            observation_spec (list): The observation specification, forwarded to :class:`IsaacSim`.
            additional_data_spec (list): The additional data specification, forwarded to :class:`IsaacSim`.
                The entries needed by the domain randomization are appended to it.
            collision_groups (list): The collision groups specification, forwarded to :class:`IsaacSim`.
            num_envs (int): Number of parallel environments.
            horizon (int): The maximum horizon for the environment.
            domain_randomization (bool): Whether the domain randomization is enabled.
            camera_position (tuple, None): The position of the camera looking at the scene, defaulting to a
                view of env 0.
            camera_target (tuple, None): The point the camera looks at, defaulting to env 0's position.
            default_joint_max_vel (torch.tensor, None): The nominal maximum velocity of every controlled
                joint, overriding the one reported by the simulation.
            nominal_p_gain (float): The proportional gain of the PD control law, before randomization.
            nominal_d_gain (float): The derivative gain of the PD control law, before randomization.
            nominal_scaling_factor (float): The factor the action is scaled by before randomization, which
                also sets the bounds of the action space.
            reward_weights (dict, None): Overrides for the coefficients the reward terms are weighed by, keyed
                like the info dictionary ``reward`` returns. Only the given keys are overridden.
            reward_params (dict, None): Overrides for the thresholds and targets the optional reward terms are
                shaped by. Only the given keys are overridden; an unknown one raises.
            clamp_reward (bool): Whether the total reward is clamped to be non-negative, as in Rudin et al.
            randomization_params (QuadrupedRandomizationParams, None): The randomization ranges, forwarded to
                :class:`QuadrupedRandomizer`.
            observed_randomization (tuple): The names of the randomized parameters the agent is told about
                through an observation. Every parameter is hidden from it by default.
            max_command_ranges (dict, None): The widest velocity command ranges the environment will ever
                sample from, keyed ``lin_vel_x``, ``lin_vel_y``, ``ang_vel_z`` and ``heading``. They bound the
                command observation and every range :meth:`command_ranges` can be set to.
            command_ranges (dict, None): The velocity command ranges to start sampling from, keyed like
                ``max_command_ranges`` and bounded by them.
            tracking_stds (dict, None): Overrides for the tolerance of the two command tracking reward terms,
                keyed ``lin_vel`` and ``ang_vel`` for the tightest tolerance, and ``lin_vel_slope`` and
                ``ang_vel_slope`` for how much it widens with the magnitude of the command. Only the given
                keys are overridden.
            command_dead_zone (float): Linear velocity commands whose norm falls below this are set to zero.
            command_resampling_time_range (tuple, None): The range, in seconds, the time until an environment
                resamples its command is drawn from. ``None`` resamples with a fixed per-step probability.
            heading_control_stiffness (float): The gain turning the error on the heading target into the yaw
                rate command.
            rel_heading_envs (float): The fraction of environments whose yaw rate command tracks a heading
                target.
            rel_standing_envs (float): The fraction of environments commanded to stand still.
            frac_rotating_envs (float): The fraction of the moving environments commanded to rotate in place.
            frac_low_speed_envs (float): The fraction of the moving environments commanded to move below
                ``low_speed_threshold``.
            low_speed_threshold (float): The velocity below which a command counts as a low speed one.

        """
        device = TorchUtils.get_device()

        self._action_spec = action_spec
        self._default_joint_angles = default_joint_angles
        self._default_joint_max_vel = default_joint_max_vel
        self._foot_bodies = foot_bodies
        self._domain_randomization = domain_randomization
        self._nominal_p_gain = nominal_p_gain
        self._nominal_d_gain = nominal_d_gain
        self._nominal_scaling_factor = nominal_scaling_factor
        self._observed_randomization = observed_randomization

        self._max_command_ranges = dict(lin_vel_x=(-1., 1.), lin_vel_y=(-1., 1.), ang_vel_z=(-math.pi, math.pi),
                                        heading=(-3.14, 3.14))
        self._max_command_ranges |= max_command_ranges or {}

        self._command_ranges = dict(lin_vel_x=(-1., 1.), lin_vel_y=(-1., 1.), ang_vel_z=(-1., 1.),
                                    heading=(-3.14, 3.14))
        self._command_ranges |= command_ranges or {}
        self._check_command_ranges(self._command_ranges)

        self._command_dead_zone = command_dead_zone
        self._command_resampling_time_range = command_resampling_time_range
        self._heading_control_stiffness = heading_control_stiffness
        self._rel_heading_envs = rel_heading_envs
        self._rel_standing_envs = rel_standing_envs
        self._frac_rotating_envs = frac_rotating_envs
        self._frac_low_speed_envs = frac_low_speed_envs
        self._low_speed_threshold = low_speed_threshold

        self._tracking_stds = dict(lin_vel=0.5, lin_vel_slope=0., ang_vel=0.5, ang_vel_slope=0.)
        self._tracking_stds |= tracking_stds or {}

        self._reward_weights = dict(
            tracking_lin_vel=1.0, tracking_ang_vel=0.5, lin_vel_z=-2.0, ang_vel_xy=-0.05, torques=-0.0002,
            joint_acc=-2.5e-7, feet_air_time=1.0, collision=-1.0, action_rate=-0.01, joint_pos_limits=-10.0
        )
        self._optional_reward_terms = (
            "flat_orientation", "joint_vel_limits", "power_draw", "similar_to_default",
            "stand_still_deviation", "base_height", "feet_air_time_high", "feet_air_time_low",
            "feet_clearance", "feet_clearance_lateral", "feet_slide", "feet_slide_low", "feet_z_velocity",
            "feet_air_time_symmetry", "long_contact", "stand_still_short_contact"
        )
        self._foot_reward_terms = (
            "feet_air_time_high", "feet_air_time_low", "feet_clearance", "feet_clearance_lateral",
            "feet_slide", "feet_slide_low", "feet_z_velocity", "feet_air_time_symmetry", "long_contact",
            "stand_still_short_contact"
        )
        self._reward_weights.update({name: 0. for name in self._optional_reward_terms})
        self._reward_weights |= reward_weights or {}

        self._reward_params = dict(
            command_threshold=0.05, air_time_threshold_high=0.5, air_time_threshold_low=0.25,
            air_time_symmetry_std=0.05, clearance_target=0.03, clearance_std=0.02,
            clearance_lateral_target=0.05, clearance_lateral_std=0.02,
            clearance_lateral_command_threshold=0.3, long_contact_threshold=0.4, long_contact_ramp_power=2.,
            long_contact_ramp_cap=1., base_height_target=0.3, joint_vel_limits_soft_ratio=0.9,
            stand_still_contact_target=0.5, stand_still_ramp_power=2., stand_still_ramp_cap=0.5
        )
        if reward_params is not None:
            unknown = set(reward_params) - set(self._reward_params)
            if unknown:
                raise ValueError(f"unknown reward parameters: {sorted(unknown)}")
            self._reward_params.update(reward_params)

        self._clamp_reward = clamp_reward

        self._randomization_params = \
            QuadrupedRandomizationParams() if randomization_params is None else randomization_params

        sim_params = {
            "gpu_found_lost_aggregate_pairs_capacity": 128 * 1024,
            "gpu_total_aggregate_pairs_capacity": 128 * 1024,
            "gpu_temp_buffer_capacity": 16777216,
            "gpu_max_rigid_patch_count": 2 * 81920,
        }
        scene_params = dict(env_spacing=3.,
                            solver_pos_it_count=torch.full((num_envs, ), 4, device=device),
                            solver_vel_it_count=torch.full((num_envs, ), 0, device=device))
        viewer_params = dict(camera_position=camera_position, camera_target=camera_target)

        additional_data_spec = additional_data_spec \
            + self._get_domain_randomization_data_spec(action_spec, trunk_body, sub_bodies)

        self._tracks_foot_state = any(self._reward_weights[name] != 0. for name in self._foot_reward_terms)
        if self._tracks_foot_state:
            additional_data_spec = additional_data_spec + self._get_foot_state_data_spec(foot_bodies)

        super().__init__(usd_path, action_spec, observation_spec, num_envs, 0.99, horizon,
                         additional_data_spec=additional_data_spec, collision_groups=collision_groups,
                         actuation_type=ActuationType.EFFORT, n_intermediate_steps=4, timestep=0.005,
                         sim_params=sim_params, scene_params=scene_params, viewer_params=viewer_params)

        self._randomizer = self._build_randomizer()
        self._observation_helper.write_data("max_joint_vel", self._randomizer.joint_max_vel,
                                            reapply_after_reset=True)

        if domain_randomization:
            all_indices = torch.arange(0, num_envs, 1, device=device)
            for name, value in self._randomizer.resample_startup(all_indices).items():
                self._observation_helper.write_data(name, value, all_indices, True)

        self._commands = torch.zeros(num_envs, 4, dtype=torch.float, device=device)
        self._is_heading_env = torch.ones((num_envs, ), dtype=torch.bool, device=device)
        self._is_standing_env = torch.zeros((num_envs, ), dtype=torch.bool, device=device)
        self._time_to_resample = torch.zeros((num_envs, ), device=device)
        self._actions = torch.zeros((num_envs, len(action_spec)), device=device)
        self._feet_air_time = torch.zeros((num_envs, len(foot_bodies)), device=device)
        self._last_actions = torch.zeros((num_envs, len(action_spec)), device=device)
        self._last_joint_vel = torch.zeros((num_envs, len(action_spec)), device=device)
        self._last_contacts = torch.zeros((num_envs, len(foot_bodies)), device=device, dtype=torch.bool)
        self._foot_air_time = torch.zeros((num_envs, len(foot_bodies)), device=device)
        self._foot_last_air_time = torch.zeros((num_envs, len(foot_bodies)), device=device)
        self._foot_contact_time = torch.zeros((num_envs, len(foot_bodies)), device=device)
        self._foot_first_contact = torch.zeros((num_envs, len(foot_bodies)), device=device, dtype=torch.bool)
        self._foot_contact = torch.zeros((num_envs, len(foot_bodies)), device=device, dtype=torch.bool)
        self._foot_positions = torch.zeros((num_envs, len(foot_bodies), 3), device=device)
        self._foot_velocities = torch.zeros((num_envs, len(foot_bodies), 3), device=device)
        self._episode_length = torch.zeros((num_envs, ), dtype=int, device=device)
        self._forward_vec = torch.tensor([1., 0., 0.], device=device).repeat((num_envs, 1))
        self._gravity = torch.tensor([0., 0., -1.], device=device).repeat((num_envs, 1))
        self._max_delay_steps_limit = self._randomization_params["max_delay_steps"]
        self._action_history = torch.zeros((self._max_delay_steps_limit + 1, num_envs, len(action_spec)),
                                           device=device)
        self._env_indices = torch.arange(0, num_envs, 1, device=device)

        self._extra_info_rewards = None
        self._setup_env_indices = None
        self._setup_joint_vel = None
        self._setup_joint_pos = None

    def setup(self, env_indices, obs):
        self._feet_air_time[env_indices] = 0.
        self._episode_length[env_indices] = 0
        self._action_history[:, env_indices, :] = 0

        self._foot_air_time[env_indices] = 0.
        self._foot_last_air_time[env_indices] = 0.
        self._foot_contact_time[env_indices] = 0.
        self._foot_first_contact[env_indices] = False
        self._foot_contact[env_indices] = False

        joint_pos = self._sample_setup_joint_pos(env_indices)
        joint_vel = torch.zeros((len(env_indices), len(self._action_spec)), device=TorchUtils.get_device())

        self._observation_helper.write_data("joint_pos", joint_pos, env_indices)
        self._observation_helper.write_data("joint_vel", joint_vel, env_indices)

        self._sample_setup_base_state(env_indices)

        self._setup_joint_pos = joint_pos
        self._setup_joint_vel = joint_vel
        self._setup_env_indices = env_indices

        self._last_joint_vel[env_indices] = joint_vel

        self._resample_domain_randomization(env_indices)
        self._resample_commands(env_indices)

        zero = torch.zeros(self.number, device=TorchUtils.get_device())
        self._extra_info_rewards = {
            "r_tracking_lin_vel": zero, "r_tracking_ang_vel": zero, "r_lin_vel_z": zero,
            "r_ang_vel_xy": zero, "r_torques": zero, "r_joint_acc": zero, "r_feet_air_time": zero,
            "r_collision": zero, "r_action_rate": zero, "r_joint_pos_limits": zero
        }

    # Taken from https://proceedings.mlr.press/v164/rudin22a.html, implemented in legged_gym:
    # https://github.com/leggedrobotics/legged_gym/blob/17847702f90d8227cd31cce9c920aa53a739a09a/legged_gym/envs/base/legged_robot.py
    def reward(self, obs, action, next_obs, absorbing):
        base_lin_vel = self._observation_helper.get_from_obs(next_obs, "base_lin_vel")
        base_lin_vel_xy = base_lin_vel[:, 0:2]
        base_lin_vel_z = base_lin_vel[:, 2]
        base_ang_vel = self._observation_helper.get_from_obs(next_obs, "base_ang_vel")
        base_ang_vel_xy = base_ang_vel[:, 0:2]
        base_ang_vel_z = base_ang_vel[:, 2]

        joint_vel = self._observation_helper.get_from_obs(next_obs, "joint_vel")
        joint_pos = self._observation_helper.get_from_obs(next_obs, "joint_pos")

        # ---------------------------------------------------------------------------

        w = self._reward_weights
        r_tracking_lin_vel = self._reward_tracking_lin_vel(base_lin_vel_xy) * w["tracking_lin_vel"] * self.dt
        r_tracking_ang_vel = self._reward_tracking_ang_vel(base_ang_vel_z) * w["tracking_ang_vel"] * self.dt
        r_lin_vel_z = self._reward_lin_vel_z(base_lin_vel_z) * w["lin_vel_z"] * self.dt
        r_ang_vel_xy = self._reward_ang_vel_xy(base_ang_vel_xy) * w["ang_vel_xy"] * self.dt
        r_torques = self._reward_torques(self._torques) * w["torques"] * self.dt
        r_joint_acc = self._reward_joint_acc(joint_vel) * w["joint_acc"] * self.dt
        r_feet_air_time = self._reward_feet_air_time() * w["feet_air_time"] * self.dt
        r_collision = self._reward_collision() * w["collision"] * self.dt
        r_action_rate = self._reward_action_rate(action) * w["action_rate"] * self.dt
        r_joint_pos_limits = self._reward_joint_pos_limits(joint_pos) * w["joint_pos_limits"] * self.dt

        self._extra_info_rewards = {
            "tracking_lin_vel": r_tracking_lin_vel, "tracking_ang_vel": r_tracking_ang_vel,
            "lin_vel_z": r_lin_vel_z, "ang_vel_xy": r_ang_vel_xy,
            "torques": r_torques, "joint_acc": r_joint_acc,
            "feet_air_time": r_feet_air_time, "collision": r_collision,
            "action_rate": r_action_rate, "joint_pos_limits": r_joint_pos_limits
        }

        reward = r_tracking_lin_vel + r_tracking_ang_vel + r_lin_vel_z + r_ang_vel_xy + r_torques \
            + r_joint_acc + r_feet_air_time + r_collision + r_action_rate + r_joint_pos_limits
        reward = reward + self._extra_reward_terms(next_obs)
        reward = reward + self._optional_reward_terms_value(next_obs)

        if self._clamp_reward:
            reward = torch.clamp(reward, min=0.)

        self._last_actions = action.clone().detach()
        self._last_joint_vel = joint_vel.clone().detach()

        return reward

    @staticmethod
    def wrap_to_pi(angles):
        angles %= 2 * math.pi
        angles -= 2 * math.pi * (angles > math.pi)
        return angles

    @property
    def command_ranges(self):
        """
        Returns:
            The velocity command ranges currently sampled from, keyed ``lin_vel_x``, ``lin_vel_y``,
            ``ang_vel_z`` and ``heading``. Assigning to this overrides only the given keys, which have to stay
            within the maximum ranges the environment was built with.

        """
        return dict(self._command_ranges)

    @command_ranges.setter
    def command_ranges(self, ranges):
        updated = dict(self._command_ranges)
        updated.update(ranges)
        self._check_command_ranges(updated)

        self._command_ranges = updated

    @property
    def tracking_stds(self):
        """
        Returns:
            The tolerance of the two command tracking reward terms, keyed ``lin_vel``, ``lin_vel_slope``,
            ``ang_vel`` and ``ang_vel_slope``. Assigning to this overrides only the given keys.

        """
        return dict(self._tracking_stds)

    @tracking_stds.setter
    def tracking_stds(self, stds):
        unknown = set(stds) - set(self._tracking_stds)
        if unknown:
            raise ValueError(f"unknown tracking tolerances: {sorted(unknown)}")

        self._tracking_stds.update(stds)

    @property
    def reward_weights(self):
        """
        Returns:
            The coefficients the reward terms are weighed by. Assigning to this overrides only the given keys.

        """
        return dict(self._reward_weights)

    @reward_weights.setter
    def reward_weights(self, weights):
        unknown = set(weights) - set(self._reward_weights)
        if unknown:
            raise ValueError(f"unknown reward weights: {sorted(unknown)}")

        if not self._tracks_foot_state:
            switched_on = [name for name in self._foot_reward_terms if weights.get(name, 0.) != 0.]
            if switched_on:
                raise ValueError(f"the reward terms {sorted(switched_on)} read the state of the feet, which "
                                 f"this environment does not track: give them a non-zero weight when "
                                 f"constructing it, since the data they read has to be declared by then")

        self._reward_weights.update(weights)

    @property
    def max_delay_steps(self):
        """
        Returns:
            The largest number of physics steps an action can currently be delayed by. It can be set to any
            value up to the one the environment was built with.

        """
        return self._randomization_params["max_delay_steps"]

    @max_delay_steps.setter
    def max_delay_steps(self, max_delay_steps):
        if max_delay_steps > self._max_delay_steps_limit:
            raise ValueError(f"the delay cannot exceed the {self._max_delay_steps_limit} steps the environment "
                             f"was built with, got {max_delay_steps}")

        self._randomization_params["max_delay_steps"] = max_delay_steps

    # construction-time hooks -------------------------------------------------------------------------------------

    def _extend_observation_spec(self):
        self._observation_helper.add_obs("projected_gravity", 3, -1, 1)
        ranges = self._max_command_ranges
        commands_upper = torch.tensor([max(abs(bound) for bound in ranges[name])
                                       for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z")],
                                      device=TorchUtils.get_device())
        self._observation_helper.add_obs("commands", 3, -commands_upper, commands_upper)

        self._action_position_limits = self._compute_action_position_limits()
        self._observation_helper.add_obs("actions", len(self._action_spec), -torch.inf, torch.inf)

        if self._observed_randomization:
            self._add_domain_randomization_obs_spec()

        self._noise_scale_vec = self._get_noise_scale_vec()
        self._soft_joint_pos_limits = self._get_soft_joint_pos_limit()

    def _modify_mdp_info(self, mdp_info):
        mdp_info.action_space = self._build_action_space()
        mdp_info.observation_space = self._build_observation_space(mdp_info.observation_space)

        return mdp_info

    def _compute_action_position_limits(self):
        """
        Computes the joint-position-relative action bounds: the controlled joints' position limits, offset
        by the default pose and scaled down by the nominal action scaling factor.

        Returns:
            Two tensors: the first contains the lower limit, and the second contains the upper limit.

        """
        return (self._actuation_helper.get_joint_pos_limits() - self._default_joint_angles) \
            / self._nominal_scaling_factor

    def _build_action_space(self):
        """
        Builds the action space from the joint-position-relative bounds computed by
        :meth:`_compute_action_position_limits`.

        Returns:
            The action space, as a :class:`~mushroom_rl.core.spaces.Box`.

        """
        action_limits = self._action_position_limits
        return Box(*action_limits, data_type=action_limits[0].dtype)

    def _build_observation_space(self, observation_space):
        """
        Builds the observation space by offsetting the joint position bounds to match the default-pose-relative
        observation and widening every bound by the noise the same way :meth:`_modify_observation` corrupts the
        runtime observation.

        Args:
            observation_space: the observation space computed from every registered observation, before this
                transformation.

        Returns:
            The observation space, as a :class:`~mushroom_rl.core.spaces.Box`.

        """
        obs_low, obs_high = observation_space.low, observation_space.high
        joint_pos_indices = self._observation_helper.obs_idx_map["joint_pos"]
        position_offset = self._randomization_params["position_offset"]
        obs_low[joint_pos_indices] -= self._default_joint_angles + position_offset
        obs_high[joint_pos_indices] -= self._default_joint_angles - position_offset
        new_obs_low = obs_low - self._noise_scale_vec
        new_obs_high = obs_high + self._noise_scale_vec
        return Box(new_obs_low, new_obs_high, data_type=new_obs_high.dtype)

    def _add_domain_randomization_obs_spec(self):
        """
        Registers an observation for every randomized parameter the environment was asked to expose, in the
        physical units the randomizer draws it in, bounded by the very range it draws it from.

        Raises:
            ValueError: if a name that is not a randomized parameter was asked for.

        """
        bounds = self._domain_randomization_obs_bounds()

        unknown = set(self._observed_randomization) - set(bounds)
        if unknown:
            raise ValueError(f"unknown randomized parameters to observe: {sorted(unknown)}")

        for name in self._observed_randomization:
            length, min_value, max_value = bounds[name]
            self._observation_helper.add_obs(name=name, length=length, min_value=min_value, max_value=max_value)

    def _domain_randomization_obs_bounds(self):
        """
        Returns:
            The length and the bounds of the observation exposing every randomized parameter, keyed by the
            name the parameter is drawn under. The bounds of the joint properties also cover their nominal
            value, which the randomizer leaves them at with a fixed probability.

        """
        n_joints = len(self._action_spec)
        params = self._randomization_params
        nominal_torque_limit = self._observation_helper.read_data("torque_limit")[0]
        nominal_joint_max_vel = self._nominal_joint_max_vel()
        nominal_mass = self._observation_helper.read_data("robot_mass")[0].sum().item()

        joint_nominal_position_min, joint_nominal_position_max = params["add_joint_nominal_position"]
        torque_limit_factor = params["torque_limit_factor"]
        joint_velocity_factor = params["joint_velocity_factor"]
        p_gain_min, p_gain_max = params["p_gain_scale"]
        d_gain_min, d_gain_max = params["d_gain_scale"]
        scaling_factor_min, scaling_factor_max = params["add_scaling_factor"]
        trunk_mass_min, trunk_mass_max = params["add_trunk_mass"]

        bounds = {
            "joint_nominal_position": (n_joints, self._default_joint_angles + joint_nominal_position_min,
                                       self._default_joint_angles + joint_nominal_position_max),
            "torque_limit": (n_joints, nominal_torque_limit * (1. - torque_limit_factor),
                             nominal_torque_limit * (1. + torque_limit_factor)),
            "joint_max_velocity": (n_joints, nominal_joint_max_vel * (1. - joint_velocity_factor),
                                   nominal_joint_max_vel * (1. + joint_velocity_factor)),
            "p_gain": (n_joints, self._nominal_p_gain * p_gain_min, self._nominal_p_gain * p_gain_max),
            "d_gain": (n_joints, self._nominal_d_gain * d_gain_min, self._nominal_d_gain * d_gain_max),
            "action_scaling_factor": (n_joints, self._nominal_scaling_factor + scaling_factor_min,
                                      self._nominal_scaling_factor + scaling_factor_max),
            "mass": (1, nominal_mass + trunk_mass_min, nominal_mass + trunk_mass_max)
        }
        for name in ("joint_damping", "joint_stiffness", "joint_armature", "joint_frictionloss"):
            nominal = self._observation_helper.read_data(name)[0]
            range_min, range_max = params[name]
            bounds[name] = (n_joints, torch.clamp(nominal, max=range_min), torch.clamp(nominal, min=range_max))

        return bounds

    # per-step hooks ------------------------------------------------------------------------------------------

    def _sample_setup_base_state(self, env_indices):
        """
        Draws the pose and the velocity a reset environment's trunk starts with, offsetting the spawn the
        robot was authored with, and writes them into the simulation.

        """
        device = TorchUtils.get_device()
        n_envs = len(env_indices)
        params = self._randomization_params

        offset_x, offset_y, offset_yaw = params["reset_base_pose_range"]
        base_pos = self._observation_helper.read_data("body_pos", env_indices)
        base_pos[:, 0] += torch_rand_float(-offset_x, offset_x, (n_envs, 1), device).squeeze(1)
        base_pos[:, 1] += torch_rand_float(-offset_y, offset_y, (n_envs, 1), device).squeeze(1)
        self._observation_helper.write_data("body_pos", base_pos, env_indices)

        yaw = torch_rand_float(-offset_yaw, offset_yaw, (n_envs, 1), device).squeeze(1)
        yaw_rotation = torch.zeros((n_envs, 4), device=device)
        yaw_rotation[:, 0] = torch.cos(yaw / 2)
        yaw_rotation[:, 3] = torch.sin(yaw / 2)
        base_rot = self._observation_helper.read_data("body_rot", env_indices)
        self._observation_helper.write_data("body_rot", quat_mul(yaw_rotation, base_rot), env_indices)

        velocity_range = torch.tensor(params["reset_base_velocity_range"], device=device)
        body_vel = (2 * torch.rand((n_envs, 6), device=device) - 1) * velocity_range
        self._observation_helper.write_data("body_vel", body_vel, env_indices)

    def _sample_setup_joint_pos(self, env_indices):
        """
        Samples the joint configuration a reset environment starts in: the pose the robot was authored with
        when the domain randomization is enabled, so that the randomized nominal the control law works from is
        an offset the robot starts out of, a randomly scaled pose otherwise.

        """
        if self._domain_randomization:
            return self._default_joint_angles.expand(len(env_indices), -1)

        r_factors = torch_rand_float(0.5, 1.5, (len(env_indices), len(self._action_spec)),
                                     device=TorchUtils.get_device())
        return self._default_joint_angles * r_factors

    def _step_finalize(self, env_indices):
        self._episode_length += 1

        self._resample_commands(self._environments_to_resample(env_indices))

        base_quat = self._observation_helper.read_data("body_rot")
        forward = quat_apply(base_quat, self._forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        yaw_rate = torch.clip(self._heading_control_stiffness * self.wrap_to_pi(self._commands[:, 3] - heading),
                              *self._command_ranges["ang_vel_z"])
        self._commands[:, 2] = torch.where(self._is_heading_env, yaw_rate, self._commands[:, 2])
        self._commands[self._is_standing_env, :3] = 0.

        self._push_domain_randomization(env_indices)

    def _environments_to_resample(self, env_indices):
        """
        Returns:
            The environments whose velocity command is due to be drawn again, either because their timer ran
            out or, when no resampling time range is set, because the per-step draw came up for them.

        """
        if self._command_resampling_time_range is None:
            do_resample = torch_rand_float(0., 1., (len(env_indices), 1),
                                           device=TorchUtils.get_device()).squeeze(-1) < (1. / 500.)
            do_resample *= self._episode_length[env_indices] > 50
            return env_indices[do_resample]

        self._time_to_resample[env_indices] -= self.dt
        return env_indices[self._time_to_resample[env_indices] <= 0.]

    def _push_domain_randomization(self, env_indices):
        """
        Applies the domain randomization happening at every step, which is the random push knocking the robot
        off balance. Everything else is drawn either once, when the simulation starts, or at every reset.

        """
        if self._domain_randomization:
            push_indices, push_velocities = self._randomizer.sample_disturbance(env_indices,
                                                                                self._episode_length, self.dt)
            self._push_robots(push_indices, push_velocities)

    def _resample_domain_randomization(self, env_indices):
        """
        Draws the randomized parameters of a fresh episode for the given environments, including the friction
        of the ground they walk on, and writes them into the simulation.

        """
        if self._domain_randomization:
            for name, value in self._randomizer.resample_reset(env_indices).items():
                self._observation_helper.write_data(name, value, env_indices, True)

            static_friction, dynamic_friction = self._randomizer.sample_friction(len(env_indices))
            self._scene_builder.set_robot_friction(static_friction, dynamic_friction, env_indices)

    def _resample_commands(self, env_ids):
        device = TorchUtils.get_device()
        n_envs = len(env_ids)
        ranges = self._command_ranges

        self._commands[env_ids, 0] = torch_rand_float(*ranges["lin_vel_x"], (n_envs, 1), device=device).squeeze(1)
        self._commands[env_ids, 1] = torch_rand_float(*ranges["lin_vel_y"], (n_envs, 1), device=device).squeeze(1)
        self._commands[env_ids, 3] = torch_rand_float(*ranges["heading"], (n_envs, 1), device=device).squeeze(1)

        if self._rel_heading_envs < 1.:
            self._commands[env_ids, 2] = torch_rand_float(*ranges["ang_vel_z"], (n_envs, 1),
                                                          device=device).squeeze(1)
            self._is_heading_env[env_ids] = torch_rand_float(0., 1., (n_envs, 1), device=device).squeeze(1) \
                <= self._rel_heading_envs

        self._bias_commands(env_ids)

        # set small commands to zero
        self._commands[env_ids, :2] *= \
            (torch.norm(self._commands[env_ids, :2], dim=1) > self._command_dead_zone).unsqueeze(1)

        if self._command_resampling_time_range is not None:
            self._time_to_resample[env_ids] = torch_rand_float(*self._command_resampling_time_range, (n_envs, 1),
                                                               device=device).squeeze(1)

    def _bias_commands(self, env_ids):
        """
        Skews the freshly drawn commands towards the regimes a uniform draw barely covers: standing still,
        turning on the spot, and walking slowly in an arbitrary direction. Every block is inert, and draws no
        random number at all, while the fraction driving it is zero.

        """
        device = TorchUtils.get_device()
        n_envs = len(env_ids)

        if self._rel_standing_envs > 0.:
            self._is_standing_env[env_ids] = torch_rand_float(0., 1., (n_envs, 1), device=device).squeeze(1) \
                <= self._rel_standing_envs

        moving = env_ids[torch.logical_not(self._is_standing_env[env_ids])]
        n_moving = len(moving)

        if self._frac_low_speed_envs > 0.:
            is_low_speed = torch_rand_float(0., 1., (n_moving, 1), device=device).squeeze(1) \
                <= self._frac_low_speed_envs
            low_speed = moving[is_low_speed]

            direction = torch.randn(len(low_speed), 3, device=device)
            direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
            magnitude = torch_rand_float(0., self._low_speed_threshold, (len(low_speed), 1), device=device)
            self._commands[low_speed, :3] = direction * magnitude

        if self._frac_rotating_envs > 0.:
            is_rotating = torch_rand_float(0., 1., (n_moving, 1), device=device).squeeze(1) \
                <= self._frac_rotating_envs
            rotating = moving[is_rotating]

            is_slow_turn = torch_rand_float(0., 1., (len(rotating), 1), device=device).squeeze(1) <= 0.5
            slow_turn = rotating[is_slow_turn]
            self._commands[slow_turn, 2] = torch_rand_float(
                -self._low_speed_threshold, self._low_speed_threshold, (len(slow_turn), 1), device=device
            ).squeeze(1)

            self._commands[rotating, :2] = 0.

    # observations ----------------------------------------------------------------------------------------------

    def _create_observation(self, obs):
        # update observation with values set in setup
        if self._setup_env_indices is not None:
            joint_pos_indices = self._observation_helper.obs_idx_map["joint_pos"]
            obs[self._setup_env_indices.unsqueeze(1), joint_pos_indices] = self._setup_joint_pos

            joint_vel_indices = self._observation_helper.obs_idx_map["joint_vel"]
            obs[self._setup_env_indices.unsqueeze(1), joint_vel_indices] = self._setup_joint_vel

            self._setup_env_indices = None

        # set missing observations
        rot = self._observation_helper.read_data("body_rot")
        gravity_indices = self._observation_helper.obs_idx_map["projected_gravity"]
        obs[:, gravity_indices] = quat_rotate_inverse(rot, self._gravity)

        command_indices = self._observation_helper.obs_idx_map["commands"]
        obs[:, command_indices] = self._commands[:, :3]

        action_indices = self._observation_helper.obs_idx_map["actions"]
        obs[:, action_indices] = self._actions

        lin_vel_indices = self._observation_helper.obs_idx_map["base_lin_vel"]
        lin_vel = self._observation_helper.get_from_obs(obs, "base_lin_vel")
        obs[:, lin_vel_indices] = quat_rotate_inverse(rot, lin_vel)

        ang_vel_indices = self._observation_helper.obs_idx_map["base_ang_vel"]
        ang_vel = self._observation_helper.get_from_obs(obs, "base_ang_vel")
        obs[:, ang_vel_indices] = quat_rotate_inverse(rot, ang_vel)

        return obs

    def _add_domain_randomization_observations(self, obs):
        """
        Fills in the observations telling the agent the randomized parameters it currently runs with. Leaves
        obs untouched when the environment exposes none, since no such observation is registered then.

        """
        for name in self._observed_randomization:
            obs[:, self._observation_helper.obs_idx_map[name]] = self._domain_randomization_obs_value(name)

        return obs

    def _domain_randomization_obs_value(self, name):
        """
        Returns:
            The current value of the randomized parameter the ``name`` observation exposes.

        """
        return self._randomizer.seen_parameters[name]

    def _modify_observation(self, obs):
        obs = self._add_domain_randomization_observations(obs)

        joint_pos_indices = self._observation_helper.obs_idx_map["joint_pos"]
        obs[:, joint_pos_indices] -= self._default_joint_angles + self._randomizer.position_offset

        command_indices = self._observation_helper.obs_idx_map["commands"]
        obs[:, command_indices] = self._commands[:, :3]

        obs += (2 * torch.rand_like(obs) - 1) * self._noise_scale_vec

        return obs

    def _create_info_dictionary(self, obs):
        return self._extra_info_rewards

    # control -----------------------------------------------------------------------------------------------------

    def _apply_action_delay(self, action):
        """
        Delays the action by the number of physics steps the randomizer draws, simulating actuation latency.
        The history is indexed in physics steps, so a delay can be shorter than a whole control step, the way
        a real actuator's is. Returns the action unchanged when the domain randomization is disabled.

        """
        if self._domain_randomization:
            n_delay_steps = self._randomizer.sample_latency()

            self._action_history = torch.roll(self._action_history, -1, dims=0)
            self._action_history[-1] = action

            return self._action_history[self._max_delay_steps_limit - n_delay_steps, self._env_indices]

        return action

    def _preprocess_action(self, action):
        self._actions[:] = action[:]
        return action

    def _compute_action(self, action):
        action = self._apply_action_delay(action)
        joint_vels = self._observation_helper.read_data("joint_vel")
        joint_positions = self._observation_helper.read_data("joint_pos")
        return self._compute_torque(action, joint_vels, joint_positions)

    def _compute_torque(self, action, joint_vels, joint_pos):
        """
        Converts the (possibly delayed) action and the current joint state into the torque to apply, through a
        PD law whose gains, action scaling, motor strength and torque limit are the domain-randomized ones.
        The target pose is offset by the randomized miscalibration of the joint encoders, which
        :meth:`_modify_observation` takes back out of the joint position the agent reads.

        Args:
            action (torch.tensor): the action provided at every intermediate step.
            joint_vels (torch.tensor): the current velocity of the controlled joints.
            joint_pos (torch.tensor): the current position of the controlled joints.

        Returns:
            The torque to apply to the controlled joints.

        """
        action_scaled = action * self._randomizer.scaling_factor
        target_joint_pos = self._randomizer.joint_nominal_pos + self._randomizer.position_offset + action_scaled

        self._torques = self._randomizer.p_gain * (target_joint_pos - joint_pos) \
            - self._randomizer.d_gain * joint_vels
        self._torques *= self._randomizer.motor_strength
        self._torques = torch.clip(self._torques, -self._randomizer.torque_limit, self._randomizer.torque_limit)

        return self._torques

    # reward function -----------------------------------------------------------------------------------------

    def _extra_reward_terms(self, next_obs):
        """
        Hook for a subclass to add reward terms beyond the ones common to every quadruped (e.g. a height
        penalty). Unlike the common terms, whatever this returns is not added to ``_extra_info_rewards``.
        Returns ``0.`` by default.

        """
        return 0.

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
        """
        Penalizes collisions on the robot's non-foot bodies. Robot-specific: the collision group it reads
        from differs between quadrupeds.

        """
        raise NotImplementedError

    def _foot_contacts(self):
        """
        Tells which feet are currently touching the ground. Robot-specific: the collision group holding the
        feet, and where in it they sit, differ between quadrupeds.

        Returns:
            A boolean tensor of shape (n_envs, n_feet).

        """
        raise NotImplementedError

    def _reward_joint_pos_limits(self, joint_pos):
        # Penalize joint positions too close to the limit
        out_of_limits = -(joint_pos - self._soft_joint_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (joint_pos - self._soft_joint_pos_limits[:, 1]).clip(min=0.)  # upper limit
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel(self, lin_vel_xy):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - lin_vel_xy), dim=1)
        std = (self._tracking_stds["lin_vel_slope"] * torch.norm(self._commands[:, :2], dim=1)) \
            .clamp(min=self._tracking_stds["lin_vel"])
        return torch.exp(-lin_vel_error/std**2)

    def _reward_tracking_ang_vel(self, ang_vel_z):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self._commands[:, 2] - ang_vel_z)
        std = (self._tracking_stds["ang_vel_slope"] * torch.abs(self._commands[:, 2])) \
            .clamp(min=self._tracking_stds["ang_vel"])
        return torch.exp(-ang_vel_error/std**2)

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self._foot_contacts()
        contact_filt = torch.logical_or(contact, self._last_contacts)
        self._last_contacts = contact
        first_contact = (self._feet_air_time > 0.) * contact_filt
        self._feet_air_time += self.dt
        # reward only on first contact with the ground
        rew_air_time = torch.sum((self._feet_air_time - 0.5) * first_contact, dim=1)
        rew_air_time *= torch.norm(self._commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self._feet_air_time *= ~contact_filt
        return rew_air_time

    # optional reward terms ---------------------------------------------------------------------------------

    def _optional_reward_terms_value(self, next_obs):
        """
        Computes the reward terms that are off by default, skipping every one whose weight is zero, so that a
        quadruped opting into none of them pays nothing for their existence. Adds each computed term to the
        info dictionary under its own name, the way the always-on terms are.

        The terms extend the Rudin et al. baseline the always-on terms implement with a foot timing and
        clearance subsystem (``feet_air_time_high``, ``feet_air_time_low``, ``feet_air_time_symmetry``,
        ``feet_clearance``, ``feet_clearance_lateral``, ``feet_slide``, ``feet_slide_low``,
        ``feet_z_velocity``, ``long_contact``), a standing-still subsystem (``stand_still_deviation``,
        ``stand_still_short_contact``, ``similar_to_default``), and ``flat_orientation``,
        ``joint_vel_limits``, ``power_draw`` and ``base_height``.

        Returns:
            The weighted sum of the active terms, or ``0.`` when none of them is.

        """
        weights = self._reward_weights
        active = [name for name in self._optional_reward_terms if weights[name] != 0.]

        if not active:
            return 0.

        if self._tracks_foot_state:
            self._update_foot_state()

        total = 0.
        for name in active:
            value = getattr(self, f"_reward_{name}")(next_obs) * weights[name] * self.dt
            self._extra_info_rewards[name] = value
            total = total + value

        return total

    def _reward_flat_orientation(self, next_obs):
        # Penalize a trunk that is not level
        projected_gravity = self._observation_helper.get_from_obs(next_obs, "projected_gravity")
        return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)

    def _reward_joint_vel_limits(self, next_obs):
        # Penalize joint velocities too close to the limit, one rad/s of excess per joint at most
        joint_vel = self._observation_helper.get_from_obs(next_obs, "joint_vel")
        soft_limit = self._randomizer.joint_max_vel * self._reward_params["joint_vel_limits_soft_ratio"]
        return torch.sum((torch.abs(joint_vel) - soft_limit).clip(min=0., max=1.), dim=1)

    def _reward_power_draw(self, next_obs):
        # Penalize the mechanical power drawn, which torques alone do not capture
        joint_vel = self._observation_helper.get_from_obs(next_obs, "joint_vel")
        return torch.sum(torch.abs(self._torques * joint_vel), dim=1)

    def _reward_similar_to_default(self, next_obs):
        # Penalize deviations from the nominal pose
        joint_pos = self._observation_helper.get_from_obs(next_obs, "joint_pos")
        return torch.sum(torch.abs(joint_pos - self._default_joint_angles), dim=1)

    def _reward_stand_still_deviation(self, next_obs):
        # Penalize deviations from the nominal pose, but only while standing still
        return self._reward_similar_to_default(next_obs) * self._is_standing_command()

    def _reward_base_height(self, next_obs):
        # Penalize a trunk held at the wrong height
        base_pos = self._observation_helper.get_from_obs(next_obs, "base_pos")
        return torch.square(base_pos[:, 2] - self._reward_params["base_height_target"])

    def _reward_feet_air_time_high(self, next_obs):
        # Reward long steps, while moving fast enough for long steps to be the right gait
        params = self._reward_params
        reward = torch.sum((self._foot_last_air_time - params["air_time_threshold_high"])
                           * self._foot_first_contact, dim=1)
        return reward * (self._command_norm() > max(params["command_threshold"], self._low_speed_threshold))

    def _reward_feet_air_time_low(self, next_obs):
        # Reward steps that are long for the commanded speed, while moving slowly enough that short ones are
        # acceptable, with the threshold interpolating up to the one of the high speed term
        params = self._reward_params
        command_norm = self._command_norm()

        alpha = (command_norm / self._low_speed_threshold).clamp(0., 1.)
        threshold = params["air_time_threshold_low"] \
            + alpha * (params["air_time_threshold_high"] - params["air_time_threshold_low"])

        reward = torch.sum((self._foot_last_air_time - threshold.unsqueeze(1)) * self._foot_first_contact, dim=1)
        reward = reward * (command_norm > params["command_threshold"])
        return reward * (command_norm < self._low_speed_threshold)

    def _reward_feet_air_time_symmetry(self, next_obs):
        # Penalize steps whose duration differs between the feet
        params = self._reward_params
        mean_air_time = self._foot_last_air_time.mean(dim=1, keepdim=True)
        deviation = torch.square(self._foot_last_air_time - mean_air_time)
        per_foot = 1. - torch.exp(-deviation / params["air_time_symmetry_std"]**2)

        reward = torch.sum(per_foot * self._foot_first_contact, dim=1)
        return reward * (self._command_norm() > params["command_threshold"])

    def _reward_feet_clearance(self, next_obs):
        # Reward lifting the feet to a given height while walking slowly, which the air time alone does not
        # tell apart from a foot barely clearing the ground for just as long
        params = self._reward_params
        command_norm = self._command_norm()

        reward = self._foot_clearance_value(params["clearance_target"], params["clearance_std"])
        reward = reward * (command_norm > params["command_threshold"])
        return reward * (command_norm < self._low_speed_threshold)

    def _reward_feet_clearance_lateral(self, next_obs):
        # Reward lifting the feet while walking sideways, where dragging one means stumbling
        params = self._reward_params
        reward = self._foot_clearance_value(params["clearance_lateral_target"], params["clearance_lateral_std"])
        return reward * (torch.abs(self._commands[:, 1]) > params["clearance_lateral_command_threshold"])

    def _reward_feet_slide(self, next_obs):
        # Penalize feet moving while they are on the ground
        return torch.sum(self._foot_velocities[:, :, :2].norm(dim=-1) * self._foot_contact, dim=1)

    def _reward_feet_slide_low(self, next_obs):
        # Penalize sliding further while walking slowly, where it should not happen at all
        return self._reward_feet_slide(next_obs) * (self._command_norm() < self._low_speed_threshold)

    def _reward_feet_z_velocity(self, next_obs):
        # Penalize feet landing hard
        return torch.sum(torch.square(self._foot_velocities[:, :, 2]) * self._foot_contact, dim=1)

    def _reward_long_contact(self, next_obs):
        # Penalize feet left on the ground for too long while moving
        params = self._reward_params
        excess = self._foot_contact_time - params["long_contact_threshold"]
        excess = excess.clamp(min=0., max=params["long_contact_ramp_cap"])

        penalty = excess.pow(params["long_contact_ramp_power"]).sum(dim=1)
        return penalty * (self._command_norm() > params["command_threshold"])

    def _reward_stand_still_short_contact(self, next_obs):
        # Penalize feet lifted off the ground while standing still
        params = self._reward_params
        command_norm = self._command_norm()

        shortfall = (params["stand_still_contact_target"] - self._foot_contact_time) \
            .clamp(min=0., max=params["stand_still_ramp_cap"])
        penalty = shortfall.pow(params["stand_still_ramp_power"]).sum(dim=1)

        penalty = penalty * (1. - command_norm / params["command_threshold"]).clamp(min=0., max=1.)
        return penalty * self._is_standing_command()

    # utilities -----------------------------------------------------------------------------------------------

    def _get_noise_scale_vec(self):
        v = torch.zeros((self._observation_helper.obs_length), device=TorchUtils.get_device())

        lin_vel = self._observation_helper.obs_idx_map["base_lin_vel"]
        ang_vel = self._observation_helper.obs_idx_map["base_ang_vel"]
        joint_positions = self._observation_helper.obs_idx_map["joint_pos"]
        joint_velocities = self._observation_helper.obs_idx_map["joint_vel"]
        gravity = self._observation_helper.obs_idx_map["projected_gravity"]
        commands = self._observation_helper.obs_idx_map["commands"]
        actions = self._observation_helper.obs_idx_map["actions"]

        v[lin_vel] = 0.1
        v[ang_vel] = 0.2
        v[joint_positions] = 0.01
        v[joint_velocities] = 1.5
        v[gravity] = 0.05
        v[commands[:3]] = 0
        v[actions] = 0

        return v

    def _get_soft_joint_pos_limit(self):
        soft_joint_pos_limits = torch.zeros(len(self._action_spec), 2, device=TorchUtils.get_device(),
                                            requires_grad=False)
        pos_limit = self._actuation_helper.get_joint_pos_limits()
        low = pos_limit[0]
        high = pos_limit[1]

        middle = (low + high) / 2
        r = high - low
        soft_joint_pos_limits[:, 0] = middle - 0.5 * r * 0.9
        soft_joint_pos_limits[:, 1] = middle + 0.5 * r * 0.9
        return soft_joint_pos_limits

    def _build_randomizer(self):
        """
        Builds the randomizer out of the nominal properties the live simulation reports, so that every
        parameter it holds is set from the moment it exists.

        """
        nominal_names = ("trunk_mass", "trunk_inertia", "trunk_com", "torque_limit", "joint_damping",
                         "joint_stiffness", "joint_armature", "joint_frictionloss", "robot_mass")
        nominal_values = {name: self._observation_helper.read_data(name) for name in nominal_names}
        nominal_values.update(joint_nominal_pos=self._default_joint_angles,
                              joint_max_vel=self._nominal_joint_max_vel(),
                              p_gain=self._nominal_p_gain, d_gain=self._nominal_d_gain,
                              action_scaling_factor=self._nominal_scaling_factor)

        return QuadrupedRandomizer(self._num_envs, len(self._action_spec), nominal_values,
                                   params=self._randomization_params)

    def _nominal_joint_max_vel(self):
        """
        Returns:
            The nominal maximum velocity of every controlled joint, the one the simulation reports unless the
            robot declared its own.

        """
        if self._default_joint_max_vel is None:
            return self._observation_helper.read_data("max_joint_vel")[0]

        return self._default_joint_max_vel

    def _update_foot_state(self):
        """
        Advances, once per step, the per-foot bookkeeping the optional foot reward terms read: which feet are
        on the ground, which of them have just landed, how long the air phase that ended lasted, how long each
        foot has been in contact, and where the feet are and how fast they move.

        The always-on ``feet_air_time`` term keeps its own separate bookkeeping, since it filters the contacts
        over two steps and accumulates the air time in a different order.

        """
        contact = self._foot_contacts()

        self._foot_first_contact = torch.logical_and(contact, self._foot_air_time > 0.)
        self._foot_last_air_time = torch.where(self._foot_first_contact, self._foot_air_time,
                                               self._foot_last_air_time)
        self._foot_air_time = torch.where(contact, torch.zeros_like(self._foot_air_time),
                                          self._foot_air_time + self.dt)
        self._foot_contact_time = torch.where(contact, self._foot_contact_time + self.dt,
                                              torch.zeros_like(self._foot_contact_time))
        self._foot_contact = contact

        n_feet = len(self._foot_bodies)
        self._foot_positions = torch.stack(
            [self._observation_helper.read_data(f"foot_pos_{i}") for i in range(n_feet)], dim=1
        )
        self._foot_velocities = torch.stack(
            [self._observation_helper.read_data(f"foot_lin_vel_{i}") for i in range(n_feet)], dim=1
        )

    def _foot_clearance_value(self, target, std):
        """
        Returns:
            How close every airborne foot is to the target height above its environment's ground, summed over
            the feet.

        """
        height = self._foot_positions[:, :, 2] - self._env_pos[:, 2].unsqueeze(1)
        airborne = torch.logical_not(self._foot_contact)
        return torch.sum(torch.exp(-torch.square(height - target) / std**2) * airborne, dim=1)

    def _command_norm(self):
        """
        Returns:
            The magnitude of the full velocity command, the linear and the angular part together, which is
            what the optional reward terms gate on.

        """
        return torch.norm(self._commands[:, :3], dim=1)

    def _is_standing_command(self):
        """
        Returns:
            Which environments are commanded to stand still.

        """
        return self._command_norm() < self._reward_params["command_threshold"]

    def _check_command_ranges(self, ranges):
        """
        Raises unless every command range is a known one, ordered, and contained in the maximum range the
        command observation was bounded by at construction.

        """
        unknown = set(ranges) - set(self._max_command_ranges)
        if unknown:
            raise ValueError(f"unknown command ranges: {sorted(unknown)}")

        for name, (low, high) in ranges.items():
            max_low, max_high = self._max_command_ranges[name]
            if low > high:
                raise ValueError(f"the {name} command range is empty: ({low}, {high})")
            if low < max_low or high > max_high:
                raise ValueError(f"the {name} command range ({low}, {high}) is not contained in the maximum "
                                 f"range ({max_low}, {max_high}) the environment was built with")

    def _push_robots(self, env_indices, velocities):
        extended_vels = self._observation_helper.read_data("body_vel", env_indices)
        extended_vels[:, :2] = velocities
        self._observation_helper.write_data("body_vel", extended_vels, env_indices)

    @staticmethod
    def _get_domain_randomization_data_spec(action_spec, trunk_body, sub_bodies):
        """
        Builds the additional data specification the domain randomization reads the nominal properties from and
        writes the randomized ones to.

        """
        return [
            ("trunk_mass", "", ObservationType.SUB_BODY_MASS, trunk_body),
            ("trunk_inertia", "", ObservationType.SUB_BODY_INERTIA, trunk_body),
            ("trunk_com", "", ObservationType.SUB_BODY_COM_POS, trunk_body),
            ("torque_limit", "", ObservationType.JOINT_MAX_EFFORT, action_spec),
            ("max_joint_vel", "", ObservationType.JOINT_MAX_VELOCITY, action_spec),
            ("joint_range", "", ObservationType.JOINT_MAX_POS, action_spec),
            ("joint_armature", "", ObservationType.JOINT_ARMATURES, action_spec),
            ("joint_frictionloss", "", ObservationType.JOINT_FRICTION_STATIC, action_spec),
            ("joint_damping", "", ObservationType.JOINT_GAIN_DAMPING, action_spec),
            ("joint_stiffness", "", ObservationType.JOINT_GAIN_STIFFNESS, action_spec),
            ("joint_default_pos", "", ObservationType.JOINT_DEFAULT_POS, action_spec),
            ("robot_mass", "", ObservationType.SUB_BODY_MASS, sub_bodies),
            ("body_pos", "", ObservationType.BODY_POS, None)
        ]

    @staticmethod
    def _get_foot_state_data_spec(foot_bodies):
        """
        Builds the additional data specification the optional foot reward terms read the world pose and
        velocity of every foot from.

        """
        return [entry
                for i, path in enumerate(foot_bodies)
                for entry in ((f"foot_pos_{i}", path, ObservationType.BODY_POS, None),
                              (f"foot_lin_vel_{i}", path, ObservationType.BODY_LIN_VEL, None))]
