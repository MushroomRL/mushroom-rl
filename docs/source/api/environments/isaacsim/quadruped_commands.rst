Quadruped commands
==================

The command the robot is asked to follow comes from a ``CommandGenerator``, passed to ``QuadrupedIsaac`` as
``command_generator``. The environment calls it on every reset and after every step, with the environments
involved and its observation helper, and observes and rewards the body-frame linear velocity and yaw rate it
commands. ``UniformVelocityCommands`` draws each velocity uniformly from its range, and is the default;
``EllipticVelocityCommands`` draws them from the ellipsoid inscribed in the box of the ranges.

.. automodule:: mushroom_rl.environments.isaacsim_envs.quadruped_commands
    :private-members:
