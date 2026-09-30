import math
import torch

from mushroom_rl.utils import TorchUtils
from mushroom_rl.utils.isaac_sim.torch_maths import torch_rand_float, quat_apply, wrap_to_pi


class CommandGenerator:
    """
    Base class for the generators of the command a quadruped is asked to follow.

    """
    def initialize(self, n_envs, dt):
        """
        Allocates the command of every environment.

        Args:
            n_envs (int): The number of parallel environments.
            dt (float): The duration of a control step.

        """
        self._n_envs = n_envs
        self._dt = dt

    def reset(self, env_ids, observation_helper):
        """
        Draws the command of the given environments for the episode they start.

        Args:
            env_ids (torch.tensor): The environments being reset.
            observation_helper (ObservationHelper): The observation helper of the environment, to read the
                state of the simulation from.

        """
        raise NotImplementedError

    def step(self, env_ids, observation_helper):
        """
        Updates the command of the given environments after a control step.

        Args:
            env_ids (torch.tensor): The environments that took the step.
            observation_helper (ObservationHelper): The observation helper of the environment, to read the
                state of the simulation from.

        """
        raise NotImplementedError

    @property
    def commands(self):
        """
        Returns:
            The body-frame linear velocity along x and y and the yaw rate every environment is commanded, as a
            tensor of shape ``(n_envs, 3)``.

        """
        raise NotImplementedError

    @property
    def command_bounds(self):
        """
        Returns:
            The largest absolute value each of the three commands can take, as a tensor of shape ``(3, )``.

        """
        raise NotImplementedError


class VelocityCommandGenerator(CommandGenerator):
    """
    Base class for the generators of velocity commands drawn from ranges. The yaw rate of a fraction of the
    environments tracks a heading target instead, and the drawn commands can be skewed towards standing still,
    turning on the spot and walking slowly.

    """
    def __init__(self, max_command_ranges=None, command_ranges=None, command_dead_zone=0.2,
                 command_resampling_time_range=None, heading_control_stiffness=0.5, rel_heading_envs=1.,
                 rel_standing_envs=0., frac_rotating_envs=0., frac_low_speed_envs=0., low_speed_threshold=0.5):
        """
        Constructor.

        Args:
            max_command_ranges (dict, None): The widest velocity command ranges the generator will ever sample
                from, keyed ``lin_vel_x``, ``lin_vel_y``, ``ang_vel_z`` and ``heading``. They bound the
                command and every range :attr:`command_ranges` can be set to.
            command_ranges (dict, None): The velocity command ranges to start sampling from, keyed like
                ``max_command_ranges`` and bounded by them.
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

        Raises:
            ValueError: if a command range is unknown, empty, or not contained in its maximum range.

        """
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

    def initialize(self, n_envs, dt):
        super().initialize(n_envs, dt)

        device = TorchUtils.get_device()
        self._commands = torch.zeros(n_envs, 4, dtype=torch.float, device=device)
        self._is_heading_env = torch.ones((n_envs, ), dtype=torch.bool, device=device)
        self._is_standing_env = torch.zeros((n_envs, ), dtype=torch.bool, device=device)
        self._time_to_resample = torch.zeros((n_envs, ), device=device)
        self._steps = torch.zeros((n_envs, ), dtype=int, device=device)
        self._forward_vec = torch.tensor([1., 0., 0.], device=device).repeat((n_envs, 1))

    def reset(self, env_ids, observation_helper):
        self._steps[env_ids] = 0
        self._resample(env_ids)
        self._track_heading(env_ids, observation_helper)

    def step(self, env_ids, observation_helper):
        self._steps[env_ids] += 1
        self._resample(self._environments_to_resample(env_ids))
        self._track_heading(env_ids, observation_helper)

    @property
    def commands(self):
        return self._commands[:, :3]

    @property
    def command_bounds(self):
        ranges = self._max_command_ranges
        return torch.tensor([max(abs(bound) for bound in ranges[name])
                             for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z")],
                            device=TorchUtils.get_device())

    @property
    def command_ranges(self):
        """
        Returns:
            The velocity command ranges currently sampled from, keyed ``lin_vel_x``, ``lin_vel_y``,
            ``ang_vel_z`` and ``heading``. Assigning to this overrides only the given keys, which have to stay
            within the maximum ranges the generator was built with.

        """
        return dict(self._command_ranges)

    @command_ranges.setter
    def command_ranges(self, ranges):
        updated = dict(self._command_ranges)
        updated.update(ranges)
        self._check_command_ranges(updated)

        self._command_ranges = updated

    def _resample(self, env_ids):
        """
        Draws a new command for the given environments.

        """
        device = TorchUtils.get_device()
        n_envs = len(env_ids)

        self._sample(env_ids)
        self._bias_commands(env_ids)

        # set small commands to zero
        self._commands[env_ids, :2] *= \
            (torch.norm(self._commands[env_ids, :2], dim=1) > self._command_dead_zone).unsqueeze(1)

        if self._command_resampling_time_range is not None:
            self._time_to_resample[env_ids] = torch_rand_float(*self._command_resampling_time_range, (n_envs, 1),
                                                               device=device).squeeze(1)

    def _track_heading(self, env_ids, observation_helper):
        """
        Turns the error on the heading target of the given environments tracking one into their yaw rate
        command, and zeroes the command of the given environments standing still.

        """
        base_quat = observation_helper.read_data("body_rot", env_ids)
        forward = quat_apply(base_quat, self._forward_vec[env_ids])
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        yaw_rate = torch.clip(self._heading_control_stiffness * wrap_to_pi(self._commands[env_ids, 3] - heading),
                              *self._command_ranges["ang_vel_z"])
        self._commands[env_ids, 2] = torch.where(self._is_heading_env[env_ids], yaw_rate, self._commands[env_ids, 2])
        self._commands[env_ids[self._is_standing_env[env_ids]], :3] = 0.

    def _sample(self, env_ids):
        """
        Draws the linear velocity, yaw rate and heading target of the given environments from the current
        ranges, and which of them track the heading target.

        """
        raise NotImplementedError

    def _bias_commands(self, env_ids):
        """
        Skews the freshly drawn commands towards the regimes a draw from the ranges barely covers: standing
        still, turning on the spot, and walking slowly in an arbitrary direction. Every block is inert, and draws
        no random number at all, while the fraction driving it is zero.

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

        self._bias_moving_commands(moving)

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

    def _bias_moving_commands(self, moving):
        """
        Further skews the commands of the given moving environments, between the low speed and the rotating
        bias. Does nothing by default.

        """

    def _environments_to_resample(self, env_ids):
        """
        Returns:
            The environments whose velocity command is due to be drawn again, either because their timer ran
            out or, when no resampling time range is set, because the per-step draw came up for them.

        """
        if self._command_resampling_time_range is None:
            do_resample = torch_rand_float(0., 1., (len(env_ids), 1),
                                           device=TorchUtils.get_device()).squeeze(-1) < (1. / 500.)
            do_resample *= self._steps[env_ids] > 50
            return env_ids[do_resample]

        self._time_to_resample[env_ids] -= self._dt
        return env_ids[self._time_to_resample[env_ids] <= 0.]

    def _check_command_ranges(self, ranges):
        """
        Raises unless every command range is a known one, ordered, and contained in the maximum range the
        generator was built with.

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
                                 f"range ({max_low}, {max_high}) the generator was built with")


class UniformVelocityCommands(VelocityCommandGenerator):
    """
    Class for velocity commands whose linear velocity along x and y, yaw rate and heading target are each drawn
    uniformly from their range, independently of one another.

    """
    def _sample(self, env_ids):
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


class EllipticVelocityCommands(VelocityCommandGenerator):
    """
    Class for velocity commands whose linear velocity along x and y and yaw rate are drawn uniformly from the
    ellipsoid inscribed in the box of their ranges, and whose heading target is drawn uniformly from its range.
    A fraction of the moving environments can be commanded a velocity on the surface of the ellipsoid.

    """
    def __init__(self, frac_max_speed_envs=0., **params):
        """
        Constructor.

        Args:
            frac_max_speed_envs (float): The fraction of the moving environments commanded a velocity on the
                surface of the ellipsoid.
            **params: Further parameters of :class:`VelocityCommandGenerator`.

        """
        super().__init__(**params)

        self._frac_max_speed_envs = frac_max_speed_envs

    def _sample(self, env_ids):
        device = TorchUtils.get_device()
        n_envs = len(env_ids)
        center, half_width = self._ellipsoid()

        direction = torch.randn(n_envs, 3, device=device)
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
        radius = torch_rand_float(0., 1., (n_envs, 1), device=device).pow(1. / 3.)
        self._commands[env_ids, :3] = center + radius * half_width * direction

        self._commands[env_ids, 3] = torch_rand_float(*self._command_ranges["heading"], (n_envs, 1),
                                                      device=device).squeeze(1)
        if self._rel_heading_envs < 1.:
            self._is_heading_env[env_ids] = torch_rand_float(0., 1., (n_envs, 1), device=device).squeeze(1) \
                <= self._rel_heading_envs

    def _bias_moving_commands(self, moving):
        if self._frac_max_speed_envs > 0.:
            device = TorchUtils.get_device()
            center, half_width = self._ellipsoid()

            is_max_speed = torch_rand_float(0., 1., (len(moving), 1), device=device).squeeze(1) \
                <= self._frac_max_speed_envs
            max_speed = moving[is_max_speed]

            direction = torch.randn(len(max_speed), 3, device=device)
            direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
            self._commands[max_speed, :3] = center + half_width * direction

    def _ellipsoid(self):
        """
        Returns:
            The center and the half-widths of the ellipsoid the linear velocity along x and y and the yaw rate
            are drawn from.

        """
        bounds = torch.tensor([self._command_ranges[name] for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z")],
                              device=TorchUtils.get_device())
        return bounds.mean(dim=1), (bounds[:, 1] - bounds[:, 0]) / 2.
