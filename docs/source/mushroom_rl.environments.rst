Environments
============

All environments implement the ``Environment`` interface. Some of them, e.g. grid worlds, are finite Markov Decision
Processes: they extend the ``FiniteMDP`` class, exposing the transition probability matrix, the reward matrix and the
initial state distribution, so that they can also be solved with the dynamic programming solvers.


Environments
------------

Atari
~~~~~

.. autoclass:: mushroom_rl.environments.atari.Atari
    :members:
    :private-members:
    :show-inheritance:

Car on hill
~~~~~~~~~~~

.. automodule:: mushroom_rl.environments.car_on_hill
    :members:
    :private-members:
    :show-inheritance:

DeepMind Control Suite
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: mushroom_rl.environments.dm_control_env
    :members:
    :private-members:
    :show-inheritance:

Finite MDP
~~~~~~~~~~

.. automodule:: mushroom_rl.environments.finite_mdp
    :members:
    :private-members:
    :show-inheritance:

Grid World
~~~~~~~~~~

.. automodule:: mushroom_rl.environments.grid_world
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.grid_world_van_hasselt
    :members:
    :private-members:
    :show-inheritance:

Gymnasium
~~~~~~~~~

.. automodule:: mushroom_rl.environments.gymnasium_env
    :members:
    :private-members:
    :show-inheritance:

Inverted pendulum
~~~~~~~~~~~~~~~~~

.. automodule:: mushroom_rl.environments.inverted_pendulum
    :members:
    :private-members:
    :show-inheritance:

Cart Pole
~~~~~~~~~

.. automodule:: mushroom_rl.environments.cart_pole
    :members:
    :private-members:
    :show-inheritance:

LQR
~~~

.. automodule:: mushroom_rl.environments.lqr
    :members:
    :private-members:
    :show-inheritance:

Minigrid
~~~~~~~~

.. automodule:: mushroom_rl.environments.minigrid_env
    :members:
    :private-members:
    :show-inheritance:


Mujoco
~~~~~~

.. automodule:: mushroom_rl.environments.mujoco
    :members:
    :private-members:
    :show-inheritance:

Air Hockey
^^^^^^^^^^

.. automodule:: mushroom_rl.environments.mujoco_envs.air_hockey.base
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.mujoco_envs.air_hockey.single
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.mujoco_envs.air_hockey.double
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.mujoco_envs.air_hockey.hit
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.mujoco_envs.air_hockey.defend
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.mujoco_envs.air_hockey.prepare
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.mujoco_envs.air_hockey.repel
    :members:
    :private-members:
    :show-inheritance:

Ball In A Cup
^^^^^^^^^^^^^

.. automodule:: mushroom_rl.environments.mujoco_envs.ball_in_a_cup
    :members:
    :private-members:
    :show-inheritance:

Locomotion
^^^^^^^^^^

.. automodule:: mushroom_rl.environments.mujoco_envs.ant
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.mujoco_envs.half_cheetah
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.mujoco_envs.hopper
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.mujoco_envs.walker_2d
    :members:
    :private-members:
    :show-inheritance:

Manipulation
^^^^^^^^^^^^

.. automodule:: mushroom_rl.environments.mujoco_envs.panda
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.mujoco_envs.reach
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.mujoco_envs.pick
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.mujoco_envs.push
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.mujoco_envs.peg_insertion
    :members:
    :private-members:
    :show-inheritance:


Puddle World
~~~~~~~~~~~~

.. automodule:: mushroom_rl.environments.puddle_world
    :members:
    :private-members:
    :show-inheritance:

Pybullet
~~~~~~~~

.. automodule:: mushroom_rl.environments.pybullet
    :members:
    :private-members:
    :show-inheritance:

Air Hockey
^^^^^^^^^^

.. automodule:: mushroom_rl.environments.pybullet_envs.air_hockey.base
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.pybullet_envs.air_hockey.single
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.pybullet_envs.air_hockey.hit
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.pybullet_envs.air_hockey.defend
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.pybullet_envs.air_hockey.prepare
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.pybullet_envs.air_hockey.repel
    :members:
    :private-members:
    :show-inheritance:

Segway
~~~~~~

.. automodule:: mushroom_rl.environments.segway
    :members:
    :private-members:
    :show-inheritance:

Ship steering
~~~~~~~~~~~~~

.. automodule:: mushroom_rl.environments.ship_steering
    :members:
    :private-members:
    :show-inheritance:

Isaac Sim
~~~~~~~~~

.. automodule:: mushroom_rl.environments.isaacsim_env
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.isaacsim_envs.a1_walking
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.isaacsim_envs.cartpole
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.isaacsim_envs.honey_badger_walking
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.isaacsim_envs.silver_badger_walking
    :members:
    :private-members:
    :show-inheritance:

Omni Isaac Gym
~~~~~~~~~~~~~~

.. automodule:: mushroom_rl.environments.omni_isaac_gym_env
    :members:
    :private-members:
    :show-inheritance:

Simple chain
~~~~~~~~~~~~

.. automodule:: mushroom_rl.environments.simple_chain
    :members:
    :private-members:
    :show-inheritance:

Taxi
~~~~

.. automodule:: mushroom_rl.environments.taxi
    :members:
    :private-members:
    :show-inheritance:
