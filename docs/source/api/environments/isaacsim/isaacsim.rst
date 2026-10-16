Isaac Sim interface
===================

The base class of every Isaac Sim environment. It owns the cloned scenes, steps the simulation for the number of
intermediate steps declared, and assembles the batched observation from the specification of the articulation
entities it is given, leaving the reward and the termination condition to the subclass.

.. automodule:: mushroom_rl.environments.isaacsim_env
    :private-members:

.. rubric:: Cart pole

The batched cart pole, the smallest Isaac Sim environment: a single actuated slider carrying a free pole, used to
check the interface end to end without the cost of a full robot.

.. automodule:: mushroom_rl.environments.isaacsim_envs.cartpole
