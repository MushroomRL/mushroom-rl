import math
import torch

from isaacsim.core.utils.torch.maths import torch_rand_float
from isaacsim.core.utils.torch.rotations import quat_apply, quat_rotate_inverse

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

    Holds everything that does not depend on which quadruped is being simulated: the command-tracking reward
    terms, the observation of gravity/commands/actions, the joint-position-relative action space, the
    randomized PD control law and the whole domain-randomization machinery driven by
    :class:`QuadrupedRandomizer`. A concrete quadruped supplies its own USD asset, controlled joints, default
    pose, trunk/foot/link names and collision groups, and is responsible for ``is_absorbing`` and the
    collision-dependent reward terms, since those genuinely differ between robots.

    """
    def __init__(self, usd_path, action_spec, default_joint_angles, trunk_body, foot_bodies, sub_bodies,
                 observation_spec, additional_data_spec, collision_groups, num_envs, horizon,
                 domain_randomization, camera_position, camera_target, default_joint_max_vel=None,
                 nominal_p_gain=20., nominal_d_gain=0.5, nominal_scaling_factor=0.25, reward_weights=None,
                 normalization_scales=None, randomization_params=None):
        """
        Constructor.

        Args:
            usd_path (str): Path to the usd file of the robot.
            action_spec (list): The names of the joints the agent controls.
            default_joint_angles (torch.tensor): The nominal joint configuration the robot stands in, actions
                are expressed relative to it.
            trunk_body (str): The name of the body whose mass, inertia and center of mass are randomized.
            foot_bodies (list): The prim paths of the feet, whose size is randomized.
            sub_bodies (list): The names of every body the robot mass is made of, the trunk first.
            observation_spec (list): The observation specification, forwarded to :class:`IsaacSim`.
            additional_data_spec (list): The additional data specification, forwarded to :class:`IsaacSim`.
                The entries the domain randomization needs are appended to it.
            collision_groups (list): The collision groups specification, forwarded to :class:`IsaacSim`.
            num_envs (int): Number of parallel environments.
            horizon (int): The maximum horizon for the environment.
            domain_randomization (bool): Whether the domain randomization is enabled. The nominal control
                parameters are set up either way, only their perturbation is switched off.
            camera_position (tuple): The position of the camera looking at the scene.
            camera_target (tuple): The point the camera looking at the scene points to.
            default_joint_max_vel (torch.tensor, None): The nominal maximum velocity of every controlled
                joint, overriding the one the simulation reports when given.
            nominal_p_gain (float): The proportional gain of the PD control law, before randomization.
            nominal_d_gain (float): The derivative gain of the PD control law, before randomization.
            nominal_scaling_factor (float): The factor the action is scaled by before randomization, which
                also sets the bounds of the action space, since actions are relative joint positions.
            reward_weights (dict, None): Overrides for the coefficients ``reward`` weighs its terms by,
                keyed the same way as the info dictionary ``reward`` returns: ``tracking_lin_vel`` (default
                ``1.0``), ``tracking_ang_vel`` (``0.5``), ``lin_vel_z`` (``-2.0``), ``ang_vel_xy`` (``-0.05``),
                ``torques`` (``-0.0002``), ``joint_acc`` (``-2.5e-7``), ``feet_air_time`` (``1.0``),
                ``collision`` (``-1.0``), ``action_rate`` (``-0.01``), ``joint_pos_limits`` (``-10.0``). Only
                the given keys are overridden; a subclass adding its own reward terms (e.g. a height penalty)
                may extend this with further keys, read back from ``self._reward_weights`` in
                :meth:`_extra_reward_terms`.
            normalization_scales (dict, None): Overrides for the values the observations of the randomized
                parameters are divided by, to bring them to a comparable range. Only the given keys are
                overridden.
            randomization_params (QuadrupedRandomizationParams, None): The randomization ranges, forwarded to
                :class:`QuadrupedRandomizer`, which falls back to the defaults when this is None.

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

        self._reward_weights = dict(
            tracking_lin_vel=1.0, tracking_ang_vel=0.5, lin_vel_z=-2.0, ang_vel_xy=-0.05, torques=-0.0002,
            joint_acc=-2.5e-7, feet_air_time=1.0, collision=-1.0, action_rate=-0.01, joint_pos_limits=-10.0
        )
        if reward_weights is not None:
            self._reward_weights.update(reward_weights)

        self._normalization_scales = dict(
            joint_nominal_position=4.6, torque_limit=1000.0 / 2, joint_max_velocity=35.0 / 2,
            joint_damping=10.0 / 2, joint_stiffness=30.0 / 2, joint_armature=0.2 / 2,
            joint_frictionloss=1.2 / 2, p_gain=100.0 / 2, d_gain=2.0 / 2, action_scaling_factor=0.8 / 2,
            mass=170.0 / 2
        )
        if normalization_scales is not None:
            self._normalization_scales.update(normalization_scales)

        self._randomization_params = \
            QuadrupedRandomizationParams() if randomization_params is None else randomization_params

        physics_material_spec = self._get_values_for_physics_materials(num_envs) if domain_randomization else None
        sim_params = {
            "gpu_found_lost_aggregate_pairs_capacity": 128 * 1024,
            "gpu_total_aggregate_pairs_capacity": 128 * 1024,
            "gpu_temp_buffer_capacity": 16777216,
            "gpu_max_rigid_patch_count": 2 * 81920,
        }
        scene_params = dict(env_spacing=3., physics_material_spec=physics_material_spec,
                            solver_pos_it_count=torch.full((num_envs, ), 4, device=device),
                            solver_vel_it_count=torch.full((num_envs, ), 0, device=device))
        viewer_params = dict(camera_position=camera_position, camera_target=camera_target)

        additional_data_spec = additional_data_spec \
            + self._get_domain_randomization_data_spec(action_spec, trunk_body, foot_bodies, sub_bodies)

        super().__init__(usd_path, action_spec, observation_spec, num_envs, 0.99, horizon,
                         additional_data_spec=additional_data_spec, collision_groups=collision_groups,
                         actuation_type=ActuationType.EFFORT, n_intermediate_steps=4, timestep=0.005,
                         sim_params=sim_params, scene_params=scene_params, viewer_params=viewer_params)

        self._randomizer = self._build_randomizer()
        self._observation_helper.write_data("max_joint_vel", self._randomizer.joint_max_vel,
                                            reapply_after_reset=True)

        self._commands = torch.zeros(num_envs, 4, dtype=torch.float, device=device)
        self._actions = torch.zeros((num_envs, len(action_spec)), device=device)
        self._feet_air_time = torch.zeros((num_envs, len(foot_bodies)), device=device)
        self._last_actions = torch.zeros((num_envs, len(action_spec)), device=device)
        self._last_joint_vel = torch.zeros((num_envs, len(action_spec)), device=device)
        self._last_contacts = torch.zeros((num_envs, len(foot_bodies)), device=device, dtype=torch.bool)
        self._episode_length = torch.zeros((num_envs, ), dtype=int, device=device)
        self._forward_vec = torch.tensor([1., 0., 0.], device=device).repeat((num_envs, 1))
        self._gravity = torch.tensor([0., 0., -1.], device=device).repeat((num_envs, 1))
        max_delay_steps = self._randomization_params["max_delay_steps"]
        self._action_history = torch.zeros((max_delay_steps + 1, num_envs, len(action_spec)), device=device)

        self._extra_info_rewards = None
        self._setup_env_indices = None
        self._setup_joint_vel = None
        self._setup_joint_pos = None

    def setup(self, env_indices, obs):
        self._feet_air_time[env_indices] = 0.
        self._episode_length[env_indices] = 0
        self._action_history[:, env_indices, :] = 0

        joint_pos = self._sample_setup_joint_pos(env_indices)
        joint_vel = torch.zeros((len(env_indices), len(self._action_spec)), device=TorchUtils.get_device())

        self._observation_helper.write_data("joint_pos", joint_pos, env_indices)
        self._observation_helper.write_data("joint_vel", joint_vel, env_indices)

        body_vel = torch_rand_float(-0.5, 0.5, (len(env_indices), 6), device=TorchUtils.get_device())
        self._observation_helper.write_data("body_vel", body_vel, env_indices)

        self._setup_joint_pos = joint_pos
        self._setup_joint_vel = joint_vel
        self._setup_env_indices = env_indices

        self._last_joint_vel[env_indices] = joint_vel

        self._resample_commands(env_indices)

        zero = torch.zeros(self.number, device=TorchUtils.get_device())
        self._extra_info_rewards = {
            "r_tracking_lin_vel": zero, "r_tracking_ang_vel": zero, "r_lin_vel_z": zero,
            "r_ang_vel_xy": zero, "r_torques": zero, "r_joint_acc": zero, "r_feet_air_time": zero,
            "r_collision": zero, "r_action_rate": zero, "r_joint_pos_limits": zero
        }

    # Taken from https://proceedings.mlr.press/v164/rudin22a.html
    # Taken from legged_gym, legged_robot.py L815-L816:
    # https://github.com/leggedrobotics/legged_gym/blob/17847702f90d8227cd31cce9c920aa53a739a09a
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

        reward = torch.clamp(reward, min=0.)

        self._last_actions = action.clone().detach()
        self._last_joint_vel = joint_vel.clone().detach()

        return reward

    @staticmethod
    def wrap_to_pi(angles):
        angles %= 2 * math.pi
        angles -= 2 * math.pi * (angles > math.pi)
        return angles

    # construction-time hooks -------------------------------------------------------------------------------------

    def _modify_mdp_info(self, mdp_info):
        action_limits = (self._actuation_helper.get_joint_pos_limits() - self._default_joint_angles) \
            / self._nominal_scaling_factor
        mdp_info.action_space = Box(*action_limits, data_type=action_limits[0].dtype)

        self._observation_helper.add_obs("projected_gravity", 3, -1, 1)
        commands_upper = torch.tensor([1., 1., math.pi], device=TorchUtils.get_device())
        self._observation_helper.add_obs("commands", 3, -commands_upper, commands_upper)
        self._observation_helper.add_obs("actions", len(self._action_spec), mdp_info.action_space.low,
                                         mdp_info.action_space.high)

        if self._domain_randomization:
            self._add_domain_randomization_obs_spec()

        self._normalization_obs_vec = self._get_obs_normalization_vec()
        self._noise_scale_vec = self._get_noise_scale_vec()
        self._soft_joint_pos_limits = self._get_soft_joint_pos_limit()

        obs_low, obs_high = self._observation_helper.obs_limits
        joint_pos_indices = self._observation_helper.obs_idx_map["joint_pos"]
        obs_low[joint_pos_indices] -= self._default_joint_angles
        obs_high[joint_pos_indices] -= self._default_joint_angles
        new_obs_low = obs_low * self._normalization_obs_vec - self._noise_scale_vec
        new_obs_high = obs_high * self._normalization_obs_vec + self._noise_scale_vec
        mdp_info.observation_space = Box(new_obs_low, new_obs_high, data_type=new_obs_high.dtype)

        return mdp_info

    def _add_domain_randomization_obs_spec(self):
        """
        Registers one observation per seen domain-randomized parameter, bounded by the very ranges the
        randomizer draws them from.

        """
        n_joints = len(self._action_spec)
        params = self._randomization_params
        scales = self._normalization_scales
        nominal_torque_limit = self._observation_helper.read_data("torque_limit")[0]
        nominal_joint_max_vel = self._nominal_joint_max_vel()

        joint_nominal_position_min, joint_nominal_position_max = params["add_joint_nominal_position"]
        self._observation_helper.add_obs(
            name="joint_nominal_position",
            length=n_joints,
            min_value=(self._default_joint_angles + joint_nominal_position_min) / scales["joint_nominal_position"],
            max_value=(self._default_joint_angles + joint_nominal_position_max) / scales["joint_nominal_position"]
        )
        torque_limit_factor = params["torque_limit_factor"]
        self._observation_helper.add_obs(
            name="torque_limit",
            length=n_joints,
            min_value=(nominal_torque_limit * (1 - torque_limit_factor)) / scales["torque_limit"] - 1.0,
            max_value=(nominal_torque_limit * (1. + torque_limit_factor)) / scales["torque_limit"] - 1.0
        )
        joint_velocity_factor = params["joint_velocity_factor"]
        self._observation_helper.add_obs(
            name="joint_max_velocity",
            length=n_joints,
            min_value=(nominal_joint_max_vel * (1 - joint_velocity_factor)) / scales["joint_max_velocity"] - 1.0,
            max_value=(nominal_joint_max_vel * (1 + joint_velocity_factor)) / scales["joint_max_velocity"] - 1.0
        )
        joint_damping_min, joint_damping_max = params["joint_damping"]
        self._observation_helper.add_obs(
            name="joint_damping",
            length=n_joints,
            min_value=joint_damping_min / scales["joint_damping"] - 1.0,
            max_value=joint_damping_max / scales["joint_damping"] - 1.0
        )
        joint_stiffness_min, joint_stiffness_max = params["joint_stiffness"]
        self._observation_helper.add_obs(
            name="joint_stiffness",
            length=n_joints,
            min_value=joint_stiffness_min / scales["joint_stiffness"] - 1.0,
            max_value=joint_stiffness_max / scales["joint_stiffness"] - 1.0
        )
        joint_armature_min, joint_armature_max = params["joint_armature"]
        self._observation_helper.add_obs(
            name="joint_armature",
            length=n_joints,
            min_value=joint_armature_min / scales["joint_armature"] - 1.0,
            max_value=joint_armature_max / scales["joint_armature"] - 1.0
        )
        joint_frictionloss_min, joint_frictionloss_max = params["joint_frictionloss"]
        self._observation_helper.add_obs(
            name="joint_frictionloss",
            length=n_joints,
            min_value=joint_frictionloss_min / scales["joint_frictionloss"] - 1.0,
            max_value=joint_frictionloss_max / scales["joint_frictionloss"] - 1.0
        )
        add_p_gain_min, add_p_gain_max = params["add_p_gain"]
        self._observation_helper.add_obs(
            name="p_gain",
            length=n_joints,
            min_value=self._nominal_p_gain + add_p_gain_min / scales["p_gain"] - 1.0,
            max_value=self._nominal_p_gain + add_p_gain_max / scales["p_gain"] - 1.0
        )
        add_d_gain_min, add_d_gain_max = params["add_d_gain"]
        self._observation_helper.add_obs(
            name="d_gain",
            length=n_joints,
            min_value=self._nominal_d_gain + add_d_gain_min / scales["d_gain"] - 1.0,
            max_value=self._nominal_d_gain + add_d_gain_max / scales["d_gain"] - 1.0
        )
        add_scaling_factor_min, add_scaling_factor_max = params["add_scaling_factor"]
        self._observation_helper.add_obs(
            name="action_scaling_factor",
            length=n_joints,
            min_value=self._nominal_scaling_factor
            + add_scaling_factor_min / scales["action_scaling_factor"] - 1.0,
            max_value=self._nominal_scaling_factor
            + add_scaling_factor_max / scales["action_scaling_factor"] - 1.0
        )
        self._observation_helper.add_obs(
            name="mass",
            length=1,
            min_value=-torch.inf,
            max_value=torch.inf
        )

    def _get_obs_normalization_vec(self):
        """
        Builds the vector every observation is scaled by, before the noise is added, to bring it to a
        comparable range.

        Returns:
            A tensor of shape (obs_length, ).

        """
        v = torch.ones((self._observation_helper.obs_length), device=TorchUtils.get_device())

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

        if self._domain_randomization:
            for name, scale in self._normalization_scales.items():
                v[self._observation_helper.obs_idx_map[name]] = 1. / scale

        return v

    # per-step hooks ------------------------------------------------------------------------------------------

    def _sample_setup_joint_pos(self, env_indices):
        """
        Samples the joint configuration a reset environment starts in: the seen nominal pose when the domain
        randomization is enabled, a randomly scaled nominal pose otherwise.

        """
        if self._domain_randomization:
            return self._randomizer.joint_nominal_pos[env_indices]

        r_factors = torch_rand_float(0.5, 1.5, (len(env_indices), len(self._action_spec)),
                                     device=TorchUtils.get_device())
        return self._default_joint_angles * r_factors

    def _step_finalize(self, env_indices):
        self._episode_length += 1

        do_resample = torch_rand_float(0., 1., (len(env_indices), 1),
                                       device=TorchUtils.get_device()).squeeze(-1) < (1. / 500.)
        do_resample *= self._episode_length[env_indices] > 50
        env_ids = env_indices[do_resample]
        self._resample_commands(env_ids)

        base_quat = self._observation_helper.read_data("body_rot")
        forward = quat_apply(base_quat, self._forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self._commands[:, 2] = torch.clip(0.5 * self.wrap_to_pi(self._commands[:, 3] - heading), -1., 1.)

        self._push_domain_randomization(env_indices)

    def _push_domain_randomization(self, env_indices):
        """
        Applies the domain randomization happening at every step: the random push knocking the robot off
        balance, the occasional switch of the actuation latency regime, and the occasional redraw of every
        randomized parameter.

        """
        if self._domain_randomization:
            push_indices, push_velocities = self._randomizer.sample_disturbance(env_indices, self._episode_length)
            self._push_robots(push_indices, push_velocities)

            self._randomizer.resample_latency()

            if self._randomizer.sample_resampling():
                all_indices = torch.arange(0, self.number, 1, device=TorchUtils.get_device())
                for name, value in self._randomizer.resample(all_indices).items():
                    self._observation_helper.write_data(name, value, all_indices, True)

    def _resample_commands(self, env_ids):
        self._commands[env_ids, 0] = torch_rand_float(-1., 1., (len(env_ids), 1),
                                                      device=TorchUtils.get_device()).squeeze(1)
        self._commands[env_ids, 1] = torch_rand_float(-1., 1., (len(env_ids), 1),
                                                      device=TorchUtils.get_device()).squeeze(1)
        self._commands[env_ids, 3] = torch_rand_float(-3.14, 3.14, (len(env_ids), 1),
                                                      device=TorchUtils.get_device()).squeeze(1)

        # set small commands to zero
        self._commands[env_ids, :2] *= (torch.norm(self._commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

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
        Fills in the observations telling the agent the seen domain-randomized parameters it currently runs
        with. Leaves obs untouched when the domain randomization is disabled, since no such observation is
        registered then.

        """
        if self._domain_randomization:
            for name, value in self._randomizer.seen_parameters.items():
                obs[:, self._observation_helper.obs_idx_map[name]] = value

        return obs

    def _modify_observation(self, obs):
        obs = self._add_domain_randomization_observations(obs)

        joint_pos_indices = self._observation_helper.obs_idx_map["joint_pos"]
        obs[:, joint_pos_indices] -= self._default_joint_angles

        command_indices = self._observation_helper.obs_idx_map["commands"]
        obs[:, command_indices] = self._commands[:, :3]

        obs *= self._normalization_obs_vec
        obs += (2 * torch.rand_like(obs) - 1) * self._noise_scale_vec

        obs = torch.clamp(obs, max=100., min=-100.)

        return obs

    def _create_info_dictionary(self, obs):
        return self._extra_info_rewards

    # control -----------------------------------------------------------------------------------------------------

    def _apply_action_delay(self, action):
        """
        Delays the action by the number of steps the randomizer draws, simulating actuation latency. Returns
        the action unchanged when the domain randomization is disabled.

        """
        if self._domain_randomization:
            n_delay_steps = self._randomizer.sample_latency()

            self._action_history = torch.roll(self._action_history, -1, dims=0)
            self._action_history[-1] = action

            return self._action_history[-1 - n_delay_steps]

        return action

    def _preprocess_action(self, action):
        action = torch.clip(action, min=-100., max=100.)
        action = self._apply_action_delay(action)
        self._actions[:] = action[:]
        return action

    def _compute_action(self, action):
        joint_vels = self._observation_helper.read_data("joint_vel")
        joint_positions = self._observation_helper.read_data("joint_pos")
        return self._compute_torque(action, joint_vels, joint_positions)

    def _compute_torque(self, action, joint_vels, joint_pos):
        """
        Converts the (possibly delayed) action and the current joint state into the torque to apply, through a
        PD law whose gains, action scaling, joint position offset, motor strength and torque limit are the
        domain-randomized ones.

        Args:
            action (torch.tensor): the action provided at every intermediate step.
            joint_vels (torch.tensor): the current velocity of the controlled joints.
            joint_pos (torch.tensor): the current position of the controlled joints.

        Returns:
            The torque to apply to the controlled joints.

        """
        action_scaled = action * self._randomizer.scaling_factor
        target_joint_pos = self._randomizer.joint_nominal_pos + action_scaled

        self._torques = self._randomizer.p_gain \
            * (target_joint_pos - joint_pos + self._randomizer.position_offset) \
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

    def _reward_joint_pos_limits(self, joint_pos):
        # Penalize joint positions too close to the limit
        out_of_limits = -(joint_pos - self._soft_joint_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (joint_pos - self._soft_joint_pos_limits[:, 1]).clip(min=0.)  # upper limit
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
        """
        Rewards long steps, on first contact with the ground. Robot-specific: the collision group it reads
        from differs between quadrupeds.

        """
        raise NotImplementedError

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

        v[lin_vel] = 0.1 * 2.0
        v[ang_vel] = 0.2 * 0.25
        v[joint_positions] = 0.01 * 1.00
        v[joint_velocities] = 1.5 * 0.05
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

        return QuadrupedRandomizer(self._num_envs, len(self._action_spec), len(self._foot_bodies),
                                   nominal_values, params=self._randomization_params)

    def _nominal_joint_max_vel(self):
        """
        Returns:
            The nominal maximum velocity of every controlled joint, the one the simulation reports unless the
            robot declared its own.

        """
        if self._default_joint_max_vel is None:
            return self._observation_helper.read_data("max_joint_vel")[0]

        return self._default_joint_max_vel

    def _push_robots(self, env_indices, velocities):
        extended_vels = self._observation_helper.read_data("body_vel", env_indices)
        extended_vels[:, :2] = velocities
        self._observation_helper.write_data("body_vel", extended_vels, env_indices)

    @staticmethod
    def _get_domain_randomization_data_spec(action_spec, trunk_body, foot_bodies, sub_bodies):
        """
        Builds the additional data specification the domain randomization reads the nominal properties from and
        writes the randomized ones to.

        """
        foot_scales = [(f"foot_scale_{i}", path, ObservationType.BODY_SCALE, None)
                       for i, path in enumerate(foot_bodies)]

        return [
            ("trunk_mass", "", ObservationType.SUB_BODY_MASS, trunk_body),
            ("trunk_inertia", "", ObservationType.SUB_BODY_INERTIA, trunk_body),
            ("trunk_com", "", ObservationType.SUB_BODY_COM_POS, trunk_body),
            *foot_scales,
            ("torque_limit", "", ObservationType.JOINT_MAX_EFFORT, action_spec),
            ("max_joint_vel", "", ObservationType.JOINT_MAX_VELOCITY, action_spec),
            ("joint_range", "", ObservationType.JOINT_MAX_POS, action_spec),
            ("joint_armature", "", ObservationType.JOINT_ARMATURES, action_spec),
            ("joint_frictionloss", "", ObservationType.JOINT_FRICTION_STATIC, action_spec),
            ("joint_damping", "", ObservationType.JOINT_GAIN_DAMPING, action_spec),
            ("joint_stiffness", "", ObservationType.JOINT_GAIN_STIFFNESS, action_spec),
            ("joint_default_pos", "", ObservationType.JOINT_DEFAULT_POS, action_spec),
            ("robot_mass", "", ObservationType.SUB_BODY_MASS, sub_bodies)
        ]

    @staticmethod
    def _get_values_for_physics_materials(num_envs):
        friction_range = [0.5, 1.25]
        num_buckets = 64
        bucket_ids = torch.randint(0, num_buckets, (num_envs, ))
        friction_buckets = (friction_range[1] - friction_range[0]) * torch.rand((num_buckets, ), device='cpu') \
            + friction_range[0]

        names = [f"custom_material_{i}" for i in bucket_ids.tolist()]
        dynamic_friction = [0.5] * num_envs
        static_friction = friction_buckets[bucket_ids].tolist()
        restitution = [0.0] * num_envs

        return list(zip(names, dynamic_friction, static_friction, restitution))
