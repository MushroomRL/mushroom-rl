Quadruped interface
===================

The quadruped walking tasks, resembling the environment implemented by Rudin et al. for "Learning to Walk in
Minutes Using Massively Parallel Deep Reinforcement Learning". The agent commands a linear and an angular
velocity, and is rewarded for tracking them while keeping the robot upright and its joints away from their
limits.

``QuadrupedIsaac`` holds everything that does not depend on which quadruped is being simulated: the command-tracking
reward terms, the joint-position-relative action space, the randomized PD control law and the observations common to
every robot.

.. automodule:: mushroom_rl.environments.isaacsim_envs.quadruped
    :private-members:

.. rubric:: Reward terms beyond the baseline

Ten reward terms are always on, and reproduce the baseline of Rudin et al. On top of them ``QuadrupedIsaac``
implements a further reward set, off by default: a foot timing and clearance subsystem, a standing-still
subsystem, and terms on the trunk attitude, the joint velocity limits, the power drawn and the height the
trunk is held at. A robot opts into them by giving them a non-zero weight, and pays nothing for the ones it
leaves at zero.
