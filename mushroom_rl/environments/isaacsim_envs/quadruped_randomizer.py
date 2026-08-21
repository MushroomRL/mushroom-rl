import math
import torch

from mushroom_rl.utils.isaac_sim.torch_maths import torch_rand_float

from mushroom_rl.utils import TorchUtils


class QuadrupedRandomizationParams:
    """
    Class for defining the domain randomization parameters of a quadruped.

    .. rubric:: Disturbance, friction and latency

    The table below lists the parameters of the random pushes, the ground friction and the action latency.

    .. csv-table::
       :header: "Parameter", "Default", "Meaning"
       :widths: 30, 18, 52

       "``push_probability``", "``1/750``", "Per-step chance an environment is pushed, when no interval is set"
       "``push_interval_range``", "``None``", "Range, in seconds, between two pushes of the same environment"
       "``push_min_episode_length``", "``50``", "Steps an environment must have been alive to be pushed"
       "``push_max_velocity``", "``1.``", "Half-width of the horizontal velocity the push imparts"
       "``static_friction``", "``(0.4, 1.4)``", "Static friction of the ground the robot walks on"
       "``dynamic_friction``", "``(0.3, 1.2)``", "Dynamic friction of the ground the robot walks on"
       "``mixed_chance``", "``0.``", "Chance an episode redraws the delay at every single action"
       "``max_delay_steps``", "``4``", "Largest number of physics steps an action can be delayed by"
       "``reset_base_pose_range``", "``(0.5, 0.5, pi)``", "Half-widths of the x, y and yaw offset a reset
       robot is spawned with"
       "``reset_base_velocity_range``", "``(0., ) * 6``", "Half-widths of the six base velocity components a
       reset robot starts with"

    .. rubric:: Robot and actuation ranges

    The table below lists the ranges the properties of the robot and of its actuation are drawn from.

    .. csv-table::
       :header: "Parameter", "Default", "Meaning"
       :widths: 30, 18, 52

       "``stay_at_default_percentage``", "``1.``", "Chance the four joint properties below stay nominal
       instead of being drawn"
       "``add_trunk_mass``", "``(-2.0, 4.0)``", "Offset on the mass of the trunk"
       "``add_com_displacement``", "``(-0.05, 0.05)``", "Offset on each axis of the center of mass of the
       trunk"
       "``add_joint_nominal_position``", "``(0., 0.)``", "Offset on the nominal pose actions are relative to"
       "``torque_limit_factor``", "``0.``", "Spread of the torque limit around its nominal value"
       "``joint_velocity_factor``", "``0.``", "Spread of the maximum joint velocity"
       "``joint_damping``", "``(0.0, 0.3)``", "Damping of every joint"
       "``joint_stiffness``", "``(0.0, 0.5)``", "Stiffness of every joint"
       "``joint_armature``", "``(0.009, 0.023)``", "Armature of every joint"
       "``joint_frictionloss``", "``(0.0, 0.1)``", "Friction loss of every joint"
       "``p_gain_scale``", "``(0.85, 1.15)``", "Factor on the proportional gain"
       "``d_gain_scale``", "``(0.85, 1.15)``", "Factor on the derivative gain"
       "``add_scaling_factor``", "``(0., 0.)``", "Offset on the action scaling factor"

    .. rubric:: Unseen noise

    The table below lists the noise applied on top of the drawn values. Every entry but the last is a
    half-width around ``1``, so a value of ``0.`` leaves the property it perturbs untouched.

    .. csv-table::
       :header: "Parameter", "Default", "Meaning"
       :widths: 30, 18, 52

       "``trunk_mass_factor``", "``0.``", "Noise on the trunk mass reaching the simulation"
       "``trunk_com_factor``", "``0.``", "Noise on the trunk center of mass"
       "``joint_damping_factor``", "``0.``", "Noise on the joint damping"
       "``joint_stiffness_factor``", "``0.``", "Noise on the joint stiffness"
       "``joint_armature_factor``", "``0.``", "Noise on the joint armature"
       "``joint_frictionloss_factor``", "``0.``", "Noise on the joint friction loss"
       "``p_gain_factor``", "``0.``", "Noise on the proportional gain the control law runs on"
       "``d_gain_factor``", "``0.``", "Noise on the derivative gain the control law runs on"
       "``motor_strength_factor``", "``0.``", "Noise on the computed torque"
       "``position_offset``", "``0.04``", "Half-width of the offset corrupting the joint position read"

    """
    def __init__(self, **overrides):
        """
        Constructor.

        Args:
            **overrides: The parameters to override, named as the defaults documented above. Any other name
                raises a ``ValueError``.

        """
        self._values = self._default_values()

        unknown = set(overrides) - set(self._values)
        if unknown:
            raise ValueError(f"unknown randomization parameters: {sorted(unknown)}")

        self._values.update(overrides)

    def __getitem__(self, name):
        return self._values[name]

    def __setitem__(self, name, value):
        if name not in self._values:
            raise ValueError(f"unknown randomization parameter: {name}")

        self._values[name] = value

    def __contains__(self, name):
        return name in self._values

    @staticmethod
    def _default_values():
        """
        Returns:
            The default value of every randomization parameter, as a fresh dictionary. A subclass introducing
            further terms extends what this returns.

        """
        return dict(
            push_probability=1. / 750., push_min_episode_length=50, push_max_velocity=1.,
            push_interval_range=None,
            static_friction=(0.4, 1.4), dynamic_friction=(0.3, 1.2),
            mixed_chance=0., max_delay_steps=4,
            reset_base_pose_range=(0.5, 0.5, math.pi),
            reset_base_velocity_range=(0., 0., 0., 0., 0., 0.),
            stay_at_default_percentage=1.,
            add_trunk_mass=(-2.0, 4.0),
            add_com_displacement=(-0.05, 0.05),
            add_joint_nominal_position=(0., 0.),
            torque_limit_factor=0.,
            joint_velocity_factor=0.,
            joint_damping=(0.0, 0.3),
            joint_stiffness=(0.0, 0.5),
            joint_armature=(0.009, 0.023),
            joint_frictionloss=(0.0, 0.1),
            p_gain_scale=(0.85, 1.15),
            d_gain_scale=(0.85, 1.15),
            add_scaling_factor=(0., 0.),
            trunk_mass_factor=0.,
            trunk_com_factor=0.,
            joint_damping_factor=0.,
            joint_stiffness_factor=0.,
            joint_armature_factor=0.,
            joint_frictionloss_factor=0.,
            p_gain_factor=0.,
            d_gain_factor=0.,
            motor_strength_factor=0.,
            position_offset=0.04
        )


class QuadrupedRandomizer:
    """
    Class for sampling and storing the domain randomization parameters of every parallel environment.

    Every parameter is drawn as a value the control law and the observations work from, and some carry a
    second, hidden draw on top: the simulation is then set up with the perturbed value while the value the
    environment reports stays the first one. Every ``*_factor`` range governs one such hidden layer, and
    setting it to ``0.`` collapses the two.

    """
    def __init__(self, n_envs, n_joints, nominal_values, params=None):
        """
        Constructor.

        Args:
            n_envs (int): Number of parallel environments.
            n_joints (int): The number of controlled joints of the robot.
            nominal_values (dict): The value every randomized parameter is centered on: the properties read
                out of the live simulation, keyed by the name they are declared under in the additional data
                specification, together with the ``joint_nominal_pos``, ``joint_max_vel``, ``p_gain``,
                ``d_gain`` and ``action_scaling_factor`` the environment declares.
            params (QuadrupedRandomizationParams, None): The randomization ranges, defaulting to
                :class:`QuadrupedRandomizationParams` built with its own defaults.

        """
        device = TorchUtils.get_device()
        shape = (n_envs, n_joints)

        self._n_envs = n_envs
        self._n_joints = n_joints
        self._params = QuadrupedRandomizationParams() if params is None else params

        self._default = dict(
            trunk_mass=nominal_values["trunk_mass"][0].clone().detach(),
            trunk_inertia=nominal_values["trunk_inertia"][0].clone().detach(),
            trunk_com=nominal_values["trunk_com"][0].clone().detach(),
            torque_limit=nominal_values["torque_limit"][0].clone().detach(),
            joint_damping=nominal_values["joint_damping"][0].clone().detach(),
            joint_stiffness=nominal_values["joint_stiffness"][0].clone().detach(),
            joint_armature=nominal_values["joint_armature"][0].clone().detach(),
            joint_frictionloss=nominal_values["joint_frictionloss"][0].clone().detach(),
            joint_nominal_pos=nominal_values["joint_nominal_pos"].clone().detach(),
            joint_max_vel=nominal_values["joint_max_vel"].clone().detach(),
            p_gain=nominal_values["p_gain"],
            d_gain=nominal_values["d_gain"],
            action_scaling_factor=nominal_values["action_scaling_factor"]
        )

        self._body_masses = nominal_values["robot_mass"]
        self._seen = dict(
            joint_nominal_position=self._default["joint_nominal_pos"].repeat((n_envs, 1)),
            torque_limit=nominal_values["torque_limit"],
            joint_max_velocity=self._default["joint_max_vel"].repeat((n_envs, 1)),
            joint_damping=nominal_values["joint_damping"],
            joint_stiffness=nominal_values["joint_stiffness"],
            joint_armature=nominal_values["joint_armature"],
            joint_frictionloss=nominal_values["joint_frictionloss"],
            p_gain=torch.full(shape, self._default["p_gain"], device=device),
            d_gain=torch.full(shape, self._default["d_gain"], device=device),
            action_scaling_factor=torch.full(shape, self._default["action_scaling_factor"], device=device),
            mass=torch.sum(self._body_masses, dim=1).unsqueeze(1)
        )

        self._unseen = dict(
            p_gain=torch.full(shape, self._default["p_gain"], device=device),
            d_gain=torch.full(shape, self._default["d_gain"], device=device),
            motor_strength=torch.ones((n_envs, 1), device=device),
            position_offset=torch.zeros(shape, device=device)
        )

        self._mixed = torch.zeros((n_envs, ), dtype=torch.bool, device=device)
        self._n_delay_steps = torch.zeros((n_envs, ), dtype=torch.long, device=device)
        self._time_to_push = torch.zeros((n_envs, ), device=device)

    def resample_startup(self, env_indices):
        """
        Draws the parameters describing the robot itself, which are fixed for the whole run: its mass, its
        center of mass and its actuator gains.

        Args:
            env_indices (torch.tensor): The environments to draw for.

        Returns:
            The properties the simulation has to be set up with, keyed by the name they are declared under in
            the additional data specification.

        """
        noise = self._sample_trunk_noise(env_indices)
        noise.update(self._sample_gain_noise(env_indices))

        simulator_values = self._sample_trunk_params(env_indices, noise)
        self._sample_gain_params(env_indices, noise)

        return simulator_values

    def resample_reset(self, env_indices):
        """
        Draws the parameters that vary from episode to episode: the joint properties, the actuation and the
        action latency.

        Args:
            env_indices (torch.tensor): The environments to draw for.

        Returns:
            The properties the simulation has to be set up with, keyed by the name they are declared under in
            the additional data specification.

        """
        noise = self._sample_joint_noise(env_indices)

        simulator_values = self._sample_joint_params(env_indices, noise)
        self._sample_actuation_params(env_indices)
        self._sample_latency_regime(env_indices)

        return simulator_values

    def sample_friction(self, n_envs):
        """
        Draws the friction of the ground each of the given environments walks on for its next episode.

        Args:
            n_envs (int): How many environments to draw a friction for.

        Returns:
            The static and dynamic friction of each of them, as two tensors on the host.

        """
        shape = (n_envs, 1)
        return (torch_rand_float(*self._params["static_friction"], shape, "cpu"),
                torch_rand_float(*self._params["dynamic_friction"], shape, "cpu"))

    def sample_disturbance(self, env_indices, episode_length, dt):
        """
        Draws the random push that knocks the robot off balance, applied only to the environments that have
        been alive long enough.

        Args:
            env_indices (torch.tensor): The environments a push may be applied to.
            episode_length (torch.tensor): The number of steps every environment has been alive for.
            dt (float): The duration of a control step.

        Returns:
            The indices of the environments to push and the horizontal velocity to push each of them with.

        """
        device = TorchUtils.get_device()
        interval_range = self._params["push_interval_range"]

        if interval_range is None:
            do_push = torch_rand_float(0., 1., (len(env_indices), 1), device).squeeze(-1) \
                < self._params["push_probability"]
        else:
            self._time_to_push[env_indices] -= dt
            do_push = self._time_to_push[env_indices] <= 0.

        push_indices = env_indices[do_push]
        push_indices = push_indices[episode_length[push_indices] > self._params["push_min_episode_length"]]

        if interval_range is not None:
            self._time_to_push[env_indices[do_push]] = torch_rand_float(
                *interval_range, (int(do_push.sum()), 1), device
            ).squeeze(1)

        max_velocity = self._params["push_max_velocity"]
        velocities = torch_rand_float(-max_velocity, max_velocity, (push_indices.shape[0], 2), device)

        return push_indices, velocities

    def sample_latency(self):
        """
        Redraws the action delay of the environments in the mixed regime, leaving the others with the delay
        drawn at the start of their episode.

        Returns:
            The number of physics steps the next action is delayed by, per environment.

        """
        if torch.any(self._mixed):
            redrawn = torch.randint(0, self._params["max_delay_steps"] + 1, (self._n_envs, ),
                                    device=TorchUtils.get_device())
            self._n_delay_steps = torch.where(self._mixed, redrawn, self._n_delay_steps)

        return self._n_delay_steps

    @property
    def params(self):
        """
        Returns:
            The randomization ranges.

        """
        return self._params

    @property
    def seen_parameters(self):
        """
        Returns:
            The parameters the agent is told about, keyed by the name of the observation exposing them.

        """
        return self._seen

    @property
    def default_parameters(self):
        """
        Returns:
            The nominal value of every randomized parameter.

        """
        return self._default

    @property
    def joint_nominal_pos(self):
        """
        Returns:
            The joint configuration the actions are currently expressed relative to.

        """
        return self._seen["joint_nominal_position"]

    @property
    def joint_max_vel(self):
        """
        Returns:
            The maximum velocity every joint is currently limited to.

        """
        return self._seen["joint_max_velocity"]

    @property
    def scaling_factor(self):
        """
        Returns:
            The factor the action is currently scaled by.

        """
        return self._seen["action_scaling_factor"]

    @property
    def torque_limit(self):
        """
        Returns:
            The torque limit the control law currently clips against.

        """
        return self._seen["torque_limit"]

    @property
    def p_gain(self):
        """
        Returns:
            The unseen proportional gain the control law currently runs on.

        """
        return self._unseen["p_gain"]

    @property
    def d_gain(self):
        """
        Returns:
            The unseen derivative gain the control law currently runs on.

        """
        return self._unseen["d_gain"]

    @property
    def motor_strength(self):
        """
        Returns:
            The unseen factor the computed torque is scaled by.

        """
        return self._unseen["motor_strength"]

    @property
    def delay_steps(self):
        """
        Returns:
            The number of physics steps the next action of every environment is delayed by.

        """
        return self._n_delay_steps

    @property
    def position_offset(self):
        """
        Returns:
            The unseen offset corrupting the joint position the control law acts on.

        """
        return self._unseen["position_offset"]

    def _sample_trunk_noise(self, env_indices):
        """
        Draws the unseen multiplicative noise perturbing the mass and the center of mass, on top of the seen
        values :meth:`_sample_trunk_params` draws.

        Returns:
            The noise factors, of shape (len(env_indices), 1), keyed by the property they perturb.

        """
        return {name: self._sample_noise_factor(env_indices.shape[0], f"{name}_factor")
                for name in ("trunk_mass", "trunk_com")}

    def _sample_gain_noise(self, env_indices):
        """
        Draws the unseen multiplicative noise on the gains the control law actually runs on.

        Returns:
            The noise factors, of shape (len(env_indices), 1).

        """
        n_envs = env_indices.shape[0]

        return dict(p_gain=self._sample_noise_factor(n_envs, "p_gain_factor"),
                    d_gain=self._sample_noise_factor(n_envs, "d_gain_factor"))

    def _sample_joint_noise(self, env_indices):
        """
        Draws the unseen multiplicative noise the joint properties reaching the simulation are perturbed by.

        Returns:
            The noise factors, of shape (len(env_indices), 1), keyed by the property they perturb.

        """
        return {name: self._sample_noise_factor(env_indices.shape[0], f"{name}_factor")
                for name in ("joint_damping", "joint_stiffness", "joint_armature", "joint_frictionloss")}

    def _sample_trunk_params(self, env_indices, noise):
        """
        Draws the seen mass and center of mass of the trunk, around the nominal values the robot is authored
        with, and returns the unseen values the simulation is actually set up with.

        """
        device = TorchUtils.get_device()
        n_envs = env_indices.shape[0]

        trunk_mass = self._default["trunk_mass"] \
            + torch_rand_float(*self._params["add_trunk_mass"], (n_envs, 1), device)
        unseen_trunk_mass = trunk_mass * noise["trunk_mass"]
        unseen_trunk_inertia = self._default["trunk_inertia"] * (unseen_trunk_mass / self._default["trunk_mass"])
        self._body_masses[env_indices, 0] = trunk_mass.squeeze(1)
        self._seen["mass"] = torch.sum(self._body_masses, dim=1).unsqueeze(1)

        trunk_com = self._default["trunk_com"] \
            + torch_rand_float(*self._params["add_com_displacement"], (n_envs, 3), device)
        unseen_trunk_com = trunk_com * noise["trunk_com"]

        return {
            "trunk_mass": unseen_trunk_mass,
            "trunk_inertia": unseen_trunk_inertia.unsqueeze(1),
            "trunk_com": unseen_trunk_com.unsqueeze(1)
        }

    def _sample_gain_params(self, env_indices, noise):
        """
        Draws the seen gains of the control law, as additive offsets around their nominal values, and the
        unseen gains it is actually run with.

        """
        device = TorchUtils.get_device()
        shape = (env_indices.shape[0], self._n_joints)

        self._seen["p_gain"][env_indices] = self._default["p_gain"] \
            * torch_rand_float(*self._params["p_gain_scale"], shape, device)
        self._seen["d_gain"][env_indices] = self._default["d_gain"] \
            * torch_rand_float(*self._params["d_gain_scale"], shape, device)

        self._unseen["p_gain"][env_indices] = self._seen["p_gain"][env_indices] * noise["p_gain"]
        self._unseen["d_gain"][env_indices] = self._seen["d_gain"][env_indices] * noise["d_gain"]

    def _sample_joint_params(self, env_indices, noise):
        """
        Draws the seen values of the joint properties, and returns the unseen values the simulation is
        actually set up with. Each is either drawn from an absolute range or left nominal, with the
        probability ``stay_at_default_percentage`` sets.

        """
        device = TorchUtils.get_device()
        n_envs = env_indices.shape[0]
        n_joints = self._n_joints

        self._seen["joint_nominal_position"][env_indices] = self._default["joint_nominal_pos"] \
            + torch_rand_float(*self._params["add_joint_nominal_position"], (n_envs, n_joints), device)

        self._seen["torque_limit"][env_indices] = self._default["torque_limit"] \
            * (1 + self._sample_symmetric_offset(n_envs, "torque_limit_factor"))
        self._seen["joint_max_velocity"][env_indices] = self._default["joint_max_vel"] \
            * (1 + self._sample_symmetric_offset(n_envs, "joint_velocity_factor"))

        stay_at_default = torch_rand_float(0, 1, (n_envs, 1), device).squeeze(-1) \
            < self._params["stay_at_default_percentage"]
        default_indices = env_indices[stay_at_default]
        random_indices = env_indices[torch.logical_not(stay_at_default)]

        for name in ("joint_damping", "joint_stiffness", "joint_armature", "joint_frictionloss"):
            self._seen[name][default_indices] = self._default[name]
            self._seen[name][random_indices] = torch_rand_float(
                *self._params[name], (random_indices.shape[0], n_joints), device
            )

        unseen_values = {
            "torque_limit": self._seen["torque_limit"][env_indices],
            "max_joint_vel": self._seen["joint_max_velocity"][env_indices],
            "joint_damping": self._seen["joint_damping"][env_indices] * noise["joint_damping"],
            "joint_stiffness":
                self._seen["joint_stiffness"][env_indices] * noise["joint_stiffness"],
            "joint_armature":
                self._seen["joint_armature"][env_indices] * noise["joint_armature"],
            "joint_frictionloss":
                self._seen["joint_frictionloss"][env_indices] * noise["joint_frictionloss"]
        }

        return unseen_values

    def _sample_actuation_params(self, env_indices):
        """
        Draws how the actuation of an episode departs from the nominal one: the factor the action is scaled by,
        the strength of the motors, and the offset corrupting the joint position the controller reads.

        """
        device = TorchUtils.get_device()
        n_envs = env_indices.shape[0]
        position_offset = self._params["position_offset"]

        self._seen["action_scaling_factor"][env_indices] = self._default["action_scaling_factor"] \
            + torch_rand_float(*self._params["add_scaling_factor"], (n_envs, self._n_joints), device)

        self._unseen["motor_strength"][env_indices] = self._sample_noise_factor(n_envs, "motor_strength_factor")
        self._unseen["position_offset"][env_indices] = torch_rand_float(
            -position_offset, position_offset, (n_envs, self._n_joints), device
        )

    def _sample_latency_regime(self, env_indices):
        """
        Draws how late the actions of an episode arrive: a fixed delay for most environments, and a delay
        redrawn at every single action for the few that land in the mixed regime.

        """
        device = TorchUtils.get_device()
        n_envs = env_indices.shape[0]

        self._mixed[env_indices] = torch_rand_float(0., 1., (n_envs, 1), device).squeeze(1) \
            < self._params["mixed_chance"]
        self._n_delay_steps[env_indices] = torch.randint(0, self._params["max_delay_steps"] + 1, (n_envs, ),
                                                         device=device)

    def _sample_symmetric_offset(self, n_envs, param_name):
        """
        Draws a per-joint offset from the symmetric range the ``param_name`` half-width defines.

        """
        half_width = self._params[param_name]
        return torch_rand_float(-half_width, half_width, (n_envs, self._n_joints), TorchUtils.get_device())

    def _sample_noise_factor(self, n_envs, param_name):
        """
        Draws one multiplicative noise factor per environment, from the symmetric range around ``1`` the
        ``param_name`` half-width defines.

        """
        half_width = self._params[param_name]
        return torch_rand_float(1 - half_width, 1 + half_width, (n_envs, 1), TorchUtils.get_device())
