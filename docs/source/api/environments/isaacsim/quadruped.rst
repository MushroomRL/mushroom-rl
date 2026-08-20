Quadruped robots
================

The quadruped walking tasks, resembling the environment implemented by Rudin et al. for "Learning to Walk in
Minutes Using Massively Parallel Deep Reinforcement Learning". The agent commands a linear and an angular
velocity, and is rewarded for tracking them while keeping the robot upright and its joints away from their
limits.

.. rubric:: Interface

``QuadrupedIsaac`` holds everything that does not depend on which quadruped is being simulated: the
command-tracking reward terms, the joint-position-relative action space, the randomized PD control law and the
observations common to every robot. ``QuadrupedRandomizer`` is not an environment: it is the description of how
far every physical and control parameter may stray from its nominal value, and of which value each parallel
environment currently runs with. ``QuadrupedIsaac`` owns one, seeds it with the nominal properties read out of
the live simulation, writes back whatever it draws, and reads the current parameters off it.

.. automodule:: mushroom_rl.environments.isaacsim_envs.quadruped
    :private-members:

.. automodule:: mushroom_rl.environments.isaacsim_envs.quadruped_randomizer
    :private-members:

.. rubric:: Robots

A concrete quadruped supplies its own USD asset, controlled joints, default pose, trunk, foot and link names and
collision groups, and implements the termination condition and the collision-dependent reward terms, since those
genuinely differ between robots. ``SilverBadgerIsaac`` subclasses ``HoneyBadgerIsaac``, differing only in that
robot-specific configuration.

.. automodule:: mushroom_rl.environments.isaacsim_envs.a1

.. automodule:: mushroom_rl.environments.isaacsim_envs.go2

.. automodule:: mushroom_rl.environments.isaacsim_envs.honey_badger

.. automodule:: mushroom_rl.environments.isaacsim_envs.silver_badger

.. rubric:: Reward terms beyond the baseline

Ten reward terms are always on, and reproduce the baseline of Rudin et al. On top of them ``QuadrupedIsaac``
implements a further reward set, off by default: a foot timing and clearance subsystem, a standing-still
subsystem, and terms on the trunk attitude, the joint velocity limits, the power drawn and the height the
trunk is held at. A robot opts into them by giving them a non-zero weight, and pays nothing for the ones it
leaves at zero.

