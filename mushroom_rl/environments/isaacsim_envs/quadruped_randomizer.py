import torch

from isaacsim.core.utils.torch.maths import torch_rand_float

from mushroom_rl.utils import TorchUtils


class QuadrupedRandomizationParams:
    """
    How far every physical and control parameter of a quadruped may stray from its nominal value.

    The values are held in a dictionary rather than in one attribute each, so that a subclass can introduce
    further randomization terms by extending :meth:`_default_values`, without the constructor having to grow a
    keyword per term. Overriding an unknown name raises, so that a misspelled parameter fails loudly instead of
    silently leaving the default in place.

    Three conventions run through the names: an ``add_`` prefix marks an additive offset drawn around the
    nominal value, a ``_factor`` suffix marks a symmetric half-width applied multiplicatively, and everything
    else is an absolute ``(low, high)`` bound.

    The nominal values every range is centered on are not parameters of the randomization: they describe the
    robot and its control law, so they belong to the environment and reach the randomizer through
    ``nominal_values``.

    .. rubric:: Disturbance, cadence and latency

    .. csv-table::
       :header: "Parameter", "Default", "Meaning"
       :widths: 30, 18, 52

       "``push_probability``", "``1/750``", "Per-step chance an environment is pushed"
       "``push_min_episode_length``", "``50``", "Steps an environment must have been alive to be pushed"
       "``push_max_velocity``", "``1.``", "Half-width of the horizontal velocity the push imparts"
       "``resample_probability``", "``0.0004``", "Per-step chance every parameter below is drawn again"
       "``latency_resample_probability``", "``0.002``", "Per-step chance the actuation latency regime switches"
       "``mixed_chance``", "``0.05``", "Chance the new regime redraws the delay at every single action"
       "``max_delay_steps``", "``1``", "Largest number of steps an action can be delayed by"

    .. rubric:: Seen ranges

    The values the agent is told about through its observation, so that it can adapt to them.

    .. csv-table::
       :header: "Parameter", "Default", "Meaning"
       :widths: 30, 18, 52

       "``stay_at_default_percentage``", "``0.3``", "Chance the four joint properties below stay nominal
       instead of being drawn"
       "``add_trunk_mass``", "``(-0.8, 0.8)``", "Offset on the mass of the trunk"
       "``add_com_displacement``", "``(-0.0025, 0.0025)``", "Offset on the center of mass of the trunk"
       "``foot_scaling``", "``(0.975, 1.025)``", "Scale of every foot"
       "``add_joint_nominal_position``", "``(-0.01, 0.01)``", "Offset on the nominal pose actions are relative
       to"
       "``torque_limit_factor``", "``0.3``", "Spread of the torque limit around its nominal value"
       "``joint_velocity_factor``", "``0.15``", "Spread of the maximum joint velocity"
       "``joint_damping``", "``(0.0, 0.3)``", "Damping of every joint"
       "``joint_stiffness``", "``(0.0, 0.5)``", "Stiffness of every joint"
       "``joint_armature``", "``(0.009, 0.023)``", "Armature of every joint"
       "``joint_frictionloss``", "``(0.0, 0.1)``", "Friction loss of every joint"
       "``add_p_gain``", "``(-3.0, 3.0)``", "Offset on the proportional gain"
       "``add_d_gain``", "``(-0.1, 0.1)``", "Offset on the derivative gain"
       "``add_scaling_factor``", "``(-0.03, 0.03)``", "Offset on the action scaling factor"

    .. rubric:: Unseen noise

    The hidden mismatch stacked on top of the seen values, which the agent is never told about and can only be
    robust to. Every entry but the last is a half-width around ``1``.

    .. csv-table::
       :header: "Parameter", "Default", "Meaning"
       :widths: 30, 18, 52

       "``trunk_mass_factor``", "``0.25``", "Noise on the trunk mass reaching the simulation"
       "``trunk_com_factor``", "``0.25``", "Noise on the trunk center of mass"
       "``foot_size_factor``", "``0.03``", "Noise on the foot scale"
       "``joint_damping_factor``", "``0.5``", "Noise on the joint damping"
       "``joint_stiffness_factor``", "``0.5``", "Noise on the joint stiffness"
       "``joint_armature_factor``", "``0.5``", "Noise on the joint armature"
       "``joint_frictionloss_factor``", "``0.5``", "Noise on the joint friction loss"
       "``p_gain_factor``", "``0.25``", "Noise on the proportional gain the control law runs on"
       "``d_gain_factor``", "``0.25``", "Noise on the derivative gain the control law runs on"
       "``motor_strength_factor``", "``0.25``", "Noise on the computed torque"
       "``position_offset``", "``0.05``", "Half-width of the offset corrupting the joint position read"

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
            resample_probability=0.0004,
            latency_resample_probability=0.002, mixed_chance=0.05, max_delay_steps=1,
            stay_at_default_percentage=0.3,
            add_trunk_mass=(-0.8, 0.8),
            add_com_displacement=(-0.0025, 0.0025),
            foot_scaling=(0.975, 1.025),
            add_joint_nominal_position=(-0.01, 0.01),
            torque_limit_factor=0.3,
            joint_velocity_factor=0.15,
            joint_damping=(0.0, 0.3),
            joint_stiffness=(0.0, 0.5),
            joint_armature=(0.009, 0.023),
            joint_frictionloss=(0.0, 0.1),
            add_p_gain=(-3.0, 3.0),
            add_d_gain=(-0.1, 0.1),
            add_scaling_factor=(-0.03, 0.03),
            trunk_mass_factor=0.25,
            trunk_com_factor=0.25,
            foot_size_factor=0.03,
            joint_damping_factor=0.5,
            joint_stiffness_factor=0.5,
            joint_armature_factor=0.5,
            joint_frictionloss_factor=0.5,
            p_gain_factor=0.25,
            d_gain_factor=0.25,
            motor_strength_factor=0.25,
            position_offset=0.05
        )


class QuadrupedRandomizer:
    """
    Which randomized parameters each parallel environment currently runs with, and how to draw them again.

    The parameters live in two layers. The *seen* layer is what the agent is told about through its
    observation, so it can adapt to it. The *unseen* layer is a further, hidden mismatch stacked on top of the
    seen one, which the agent is never told about and can only be robust to: the control law runs on unseen
    gains, and the simulation is set up with unseen masses, sizes and joint properties. The two are drawn
    separately, and the pairing is not one to one, since some parameters only have a seen layer.

    Each layer is a dictionary keyed by parameter name, since that is how the environment consumes them: the
    seen layer fills the observations one for one, and :meth:`resample` returns the properties to write into
    the simulation in the same shape.

    The randomizer never touches the simulation itself. It is built once the simulation is live, out of the
    nominal properties read from it, and from then on the environment writes back whatever :meth:`resample`
    hands over and reads the current parameters off the properties below, so that the same randomizer serves
    any quadruped.

    """
    def __init__(self, n_envs, n_joints, n_feet, nominal_values, params=None):
        """
        Constructor.

        Args:
            n_envs (int): Number of parallel environments.
            n_joints (int): The number of controlled joints of the robot.
            n_feet (int): The number of feet of the robot.
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
        self._n_feet = n_feet
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

        self._mixed = False
        self._n_delay_steps = 0

    def resample(self, env_indices):
        """
        Draws every randomized parameter again, for the given environments. The unseen noise is drawn first,
        since the seen parameters are perturbed by it before reaching the simulation.

        Args:
            env_indices (torch.tensor): The environments to draw for.

        Returns:
            The properties the simulation has to be set up with, keyed by the name they are declared under in
            the additional data specification.

        """
        noise = self._sample_simulator_noise(env_indices)
        noise.update(self._sample_controller_noise(env_indices))
        simulator_values = self._sample_simulator_params(env_indices, noise)
        self._sample_controller_params(env_indices, noise)

        return simulator_values

    def sample_disturbance(self, env_indices, episode_length):
        """
        Draws the random push that knocks the robot off balance, applied only to the environments that have
        been alive long enough for a push to be meaningful.

        Args:
            env_indices (torch.tensor): The environments a push may be applied to.
            episode_length (torch.tensor): The number of steps every environment has been alive for.

        Returns:
            The indices of the environments to push and the horizontal velocity to push each of them with.

        """
        device = TorchUtils.get_device()

        do_push = torch_rand_float(0., 1., (len(env_indices), 1), device).squeeze(-1) \
            < self._params["push_probability"]
        push_indices = env_indices[do_push]
        push_indices = push_indices[episode_length[push_indices] > self._params["push_min_episode_length"]]

        max_velocity = self._params["push_max_velocity"]
        velocities = torch_rand_float(-max_velocity, max_velocity, (push_indices.shape[0], 2), device)

        return push_indices, velocities

    def sample_resampling(self):
        """
        Returns:
            Whether every randomized parameter should be drawn again at this step.

        """
        return torch.rand(()).item() < self._params["resample_probability"]

    def resample_latency(self):
        """
        Occasionally switches the actuation latency between a fixed and a "mixed" regime, in which the delay is
        redrawn at every single action.

        """
        if torch.rand(()).item() < self._params["latency_resample_probability"]:
            self._mixed = torch.rand(()).item() < self._params["mixed_chance"]
            self._n_delay_steps = 0

    def sample_latency(self):
        """
        Returns:
            The number of steps the action about to be applied is delayed by.

        """
        if self._mixed:
            self._n_delay_steps = torch.randint(0, self._params["max_delay_steps"] + 1, (1,)).item()

        return self._n_delay_steps

    @property
    def params(self):
        """
        Returns:
            The randomization ranges, so that the environment can derive the bounds of the observations
            exposing the randomized parameters from the very numbers the sampling draws from.

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
            The nominal value of every randomized parameter, the drawn ones are spread around.

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
            The proportional gain the control law currently runs on, which is the unseen one: the agent is
            told the seen gain instead, and has to be robust to the mismatch.

        """
        return self._unseen["p_gain"]

    @property
    def d_gain(self):
        """
        Returns:
            The derivative gain the control law currently runs on, unseen just like :meth:`p_gain`.

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
    def position_offset(self):
        """
        Returns:
            The unseen offset corrupting the joint position the control law acts on.

        """
        return self._unseen["position_offset"]

    def _sample_simulator_noise(self, env_indices):
        """
        Draws the unseen multiplicative noise the simulated properties are perturbed by, on top of the seen
        values :meth:`_sample_simulator_params` draws.

        Returns:
            The noise factors, of shape (len(env_indices), 1), keyed by the property they perturb.

        """
        return {name: self._sample_noise_factor(env_indices.shape[0], f"{name}_factor")
                for name in ("trunk_mass", "trunk_com", "foot_size", "joint_damping", "joint_stiffness",
                             "joint_armature", "joint_frictionloss")}

    def _sample_controller_noise(self, env_indices):
        """
        Draws the unseen perturbations of the control law: multiplicative noise on the gains and on the motor
        strength, and an additive offset on the joint position the controller believes it reads. The motor
        strength and the offset are applied directly by the control law, so they are stored rather than
        returned.

        Returns:
            The gain noise factors, of shape (len(env_indices), 1).

        """
        n_envs = env_indices.shape[0]
        position_offset = self._params["position_offset"]

        noise = dict(p_gain=self._sample_noise_factor(n_envs, "p_gain_factor"),
                     d_gain=self._sample_noise_factor(n_envs, "d_gain_factor"))

        self._unseen["motor_strength"][env_indices] = self._sample_noise_factor(n_envs, "motor_strength_factor")
        self._unseen["position_offset"][env_indices] = torch_rand_float(
            -position_offset, position_offset, (n_envs, self._n_joints), TorchUtils.get_device()
        )

        return noise

    def _sample_simulator_params(self, env_indices, noise):
        """
        Draws the seen values of the simulated properties, around the nominal values the robot is authored
        with, and returns the unseen values the simulation is actually set up with. The joint properties are
        either drawn from an absolute range or left at their nominal value, with the probability the
        ``stay_at_default_percentage`` parameter sets.

        """
        device = TorchUtils.get_device()
        n_envs = env_indices.shape[0]
        n_joints = self._n_joints

        trunk_mass = self._default["trunk_mass"] \
            + torch_rand_float(*self._params["add_trunk_mass"], (n_envs, 1), device)
        unseen_trunk_mass = trunk_mass * noise["trunk_mass"]
        unseen_trunk_inertia = self._default["trunk_inertia"] + (unseen_trunk_mass / self._default["trunk_mass"])
        self._body_masses[env_indices, 0] = trunk_mass.squeeze(1)
        self._seen["mass"] = torch.sum(self._body_masses, dim=1).unsqueeze(1)

        trunk_com = self._default["trunk_com"] \
            + torch_rand_float(*self._params["add_com_displacement"], (n_envs, 1), device)
        unseen_trunk_com = trunk_com * noise["trunk_com"]

        foot_scaling = torch_rand_float(*self._params["foot_scaling"], (n_envs, self._n_feet), device)
        unseen_foot_scaling = foot_scaling * noise["foot_size"]

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
            "trunk_mass": unseen_trunk_mass,
            "trunk_inertia": unseen_trunk_inertia.unsqueeze(1),
            "trunk_com": unseen_trunk_com.unsqueeze(1),
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
        for i in range(self._n_feet):
            unseen_values[f"foot_scale_{i}"] = unseen_foot_scaling[:, i].unsqueeze(1).repeat(1, 3)

        return unseen_values

    def _sample_controller_params(self, env_indices, noise):
        """
        Draws the seen parameters of the control law, as additive offsets around their nominal values, and the
        unseen gains the control law is actually run with.

        """
        device = TorchUtils.get_device()
        shape = (env_indices.shape[0], self._n_joints)

        self._seen["p_gain"][env_indices] = self._default["p_gain"] \
            + torch_rand_float(*self._params["add_p_gain"], shape, device)
        self._seen["d_gain"][env_indices] = self._default["d_gain"] \
            + torch_rand_float(*self._params["add_d_gain"], shape, device)
        self._seen["action_scaling_factor"][env_indices] = self._default["action_scaling_factor"] \
            + torch_rand_float(*self._params["add_scaling_factor"], shape, device)

        self._unseen["p_gain"][env_indices] = self._seen["p_gain"][env_indices] * noise["p_gain"]
        self._unseen["d_gain"][env_indices] = self._seen["d_gain"][env_indices] * noise["d_gain"]

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
