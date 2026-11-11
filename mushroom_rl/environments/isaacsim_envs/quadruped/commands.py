import math
import torch

from mushroom_rl.utils import TorchUtils
from mushroom_rl.utils.isaac_sim.torch_maths import torch_rand_float, quat_apply, wrap_to_pi


class CommandGenerator:
    """
    Base class for the generators of the command a quadruped is asked to follow.

    """
    def __init__(self):
        """
        Constructor.

        """
        self._n_envs = None
        self._dt = None

    def initialize(self, n_envs, dt):
        """
        Allocates the command of every environment.

        Args:
            n_envs (int): The number of parallel environments.
            dt (float): The duration of a control step.

        """
        self._n_envs = n_envs
        self._dt = dt

    def reset(self, env_mask, observation_helper):
        """
        Draws the command of the given environments for the episode they start.

        Args:
            env_mask (torch.tensor): A boolean mask of the environments being reset.
            observation_helper (ObservationHelper): The observation helper of the environment, to read the
                state of the simulation from.

        """
        raise NotImplementedError

    def step(self, env_mask, observation_helper):
        """
        Updates the command of the given environments after a control step.

        Args:
            env_mask (torch.tensor): A boolean mask of the environments that took the step.
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
        super().__init__()

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

        self._commands = None
        self._is_heading_env = None
        self._is_standing_env = None
        self._time_to_resample = None
        self._steps = None
        self._forward_vec = None

    def initialize(self, n_envs, dt):
        super().initialize(n_envs, dt)

        device = TorchUtils.get_device()
        self._commands = torch.zeros(n_envs, 4, dtype=torch.float, device=device)
        self._is_heading_env = torch.ones((n_envs, ), dtype=torch.bool, device=device)
        self._is_standing_env = torch.zeros((n_envs, ), dtype=torch.bool, device=device)
        self._time_to_resample = torch.zeros((n_envs, ), device=device)
        self._steps = torch.zeros((n_envs, ), dtype=int, device=device)
        self._forward_vec = torch.tensor([1., 0., 0.], device=device).repeat((n_envs, 1))

        self._update_sampling_parameters()

    def reset(self, env_mask, observation_helper):
        self._steps.masked_fill_(env_mask, 0)
        self._resample(env_mask)
        self._track_heading(env_mask, observation_helper)

    def step(self, env_mask, observation_helper):
        self._steps += env_mask
        self._resample(self._environments_to_resample(env_mask))
        self._track_heading(env_mask, observation_helper)

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
        self._update_sampling_parameters()

    def _resample(self, env_mask):
        """
        Draws a new command for the given environments.

        """
        self._sample(env_mask)
        self._bias_commands(env_mask)

        # set small commands to zero
        small = env_mask & (torch.norm(self._commands[:, :2], dim=1) <= self._command_dead_zone)
        self._commands[:, :2].masked_fill_(small.unsqueeze(1), 0.)

        if self._command_resampling_time_range is not None:
            time_to_resample = self._draw_uniform(*self._command_resampling_time_range)
            self._time_to_resample = time_to_resample.where(env_mask, self._time_to_resample)

    def _track_heading(self, env_mask, observation_helper):
        """
        Turns the error on the heading target of the given environments tracking one into their yaw rate
        command, and zeroes the command of the environments standing still.

        """
        base_quat = observation_helper.read_data("body_rot")
        forward = quat_apply(base_quat, self._forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        yaw_rate = torch.clip(self._heading_control_stiffness * wrap_to_pi(self._commands[:, 3] - heading),
                              *self._command_ranges["ang_vel_z"])
        self._commands[:, 2] = yaw_rate.where(env_mask & self._is_heading_env, self._commands[:, 2])
        self._commands[:, :3].masked_fill_(self._is_standing_env.unsqueeze(1), 0.)

    def _sample(self, env_mask):
        """
        Draws the linear velocity, yaw rate and heading target of the given environments from the current
        ranges, and which of them track the heading target.

        """
        raise NotImplementedError

    def _bias_commands(self, env_mask):
        """
        Skews the freshly drawn commands towards standing still, turning on the spot, and walking slowly in an
        arbitrary direction, according to ``rel_standing_envs``, ``frac_rotating_envs`` and ``frac_low_speed_envs``.

        """
        if self._rel_standing_envs > 0.:
            is_standing = self._draw_uniform(0., 1.) <= self._rel_standing_envs
            self._is_standing_env = is_standing.where(env_mask, self._is_standing_env)

        moving = env_mask & ~self._is_standing_env

        if self._frac_low_speed_envs > 0.:
            low_speed = moving & (self._draw_uniform(0., 1.) <= self._frac_low_speed_envs)
            command = self._draw_direction() * self._draw_uniform(0., self._low_speed_threshold).unsqueeze(1)
            self._commands[:, :3] = command.where(low_speed.unsqueeze(1), self._commands[:, :3])

        self._bias_moving_commands(moving)

        if self._frac_rotating_envs > 0.:
            rotating = moving & (self._draw_uniform(0., 1.) <= self._frac_rotating_envs)
            slow_turn = rotating & (self._draw_uniform(0., 1.) <= 0.5)
            yaw_rate = self._draw_uniform(-self._low_speed_threshold, self._low_speed_threshold)
            self._commands[:, 2] = yaw_rate.where(slow_turn, self._commands[:, 2])
            self._commands[:, :2].masked_fill_(rotating.unsqueeze(1), 0.)

    def _bias_moving_commands(self, moving):
        """
        Further skews the commands of the given moving environments, between the low speed and the rotating
        bias. Does nothing by default.

        """
        pass

    def _draw_uniform(self, low, high):
        """
        Returns:
            One value per environment, drawn uniformly between ``low`` and ``high``.

        """
        return torch_rand_float(low, high, (self._n_envs, 1), device=TorchUtils.get_device()).squeeze(1)

    def _draw_direction(self):
        """
        Returns:
            One unit vector per environment, drawn uniformly over the directions in the space of the three
            commands.

        """
        direction = torch.randn(self._n_envs, 3, device=TorchUtils.get_device())
        return direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)

    def _update_sampling_parameters(self):
        """
        Updates what a subclass derives from the command ranges, once the generator is initialized and whenever
        the ranges are set. Does nothing by default.

        """
        pass

    def _environments_to_resample(self, env_mask):
        """
        Returns:
            The mask of the environments whose velocity command is due to be drawn again, either because their
            timer ran out or, when no resampling time range is set, because the per-step draw came up for them.

        """
        if self._command_resampling_time_range is None:
            return env_mask & (self._draw_uniform(0., 1.) < 1. / 500.) & (self._steps > 50)

        self._time_to_resample -= self._dt * env_mask
        return env_mask & (self._time_to_resample <= 0.)

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
    def _sample(self, env_mask):
        ranges = self._command_ranges

        self._commands[:, 0] = self._draw_uniform(*ranges["lin_vel_x"]).where(env_mask, self._commands[:, 0])
        self._commands[:, 1] = self._draw_uniform(*ranges["lin_vel_y"]).where(env_mask, self._commands[:, 1])
        self._commands[:, 3] = self._draw_uniform(*ranges["heading"]).where(env_mask, self._commands[:, 3])

        if self._rel_heading_envs < 1.:
            self._commands[:, 2] = self._draw_uniform(*ranges["ang_vel_z"]).where(env_mask, self._commands[:, 2])

            is_heading_env = self._draw_uniform(0., 1.) <= self._rel_heading_envs
            self._is_heading_env = is_heading_env.where(env_mask, self._is_heading_env)


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

        self._center = None
        self._half_width = None

    def _sample(self, env_mask):
        direction = self._draw_direction()
        radius = self._draw_uniform(0., 1.).pow(1. / 3.).unsqueeze(1)
        command = self._center + radius * self._half_width * direction
        self._commands[:, :3] = command.where(env_mask.unsqueeze(1), self._commands[:, :3])

        heading = self._draw_uniform(*self._command_ranges["heading"])
        self._commands[:, 3] = heading.where(env_mask, self._commands[:, 3])

        if self._rel_heading_envs < 1.:
            is_heading_env = self._draw_uniform(0., 1.) <= self._rel_heading_envs
            self._is_heading_env = is_heading_env.where(env_mask, self._is_heading_env)

    def _bias_moving_commands(self, moving):
        if self._frac_max_speed_envs > 0.:
            max_speed = moving & (self._draw_uniform(0., 1.) <= self._frac_max_speed_envs)
            command = self._center + self._half_width * self._draw_direction()
            self._commands[:, :3] = command.where(max_speed.unsqueeze(1), self._commands[:, :3])

    def _update_sampling_parameters(self):
        """
        Stores the center and the half-widths of the ellipsoid the linear velocity along x and y and the yaw rate
        are drawn from.

        """
        bounds = torch.tensor([self._command_ranges[name] for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z")],
                              device=TorchUtils.get_device())
        self._center = bounds.mean(dim=1)
        self._half_width = (bounds[:, 1] - bounds[:, 0]) / 2.
