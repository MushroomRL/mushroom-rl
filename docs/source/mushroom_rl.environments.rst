Environments
============

In mushroom_rl we distinguish between two different types of environment classes:


- proper environments
- generators

While environments directly implement the ``Environment`` interface, generators
are a set of methods used to generate finite markov chains that represent a
specific environment  e.g., grid worlds.


Environments
------------

Atari
~~~~~

.. automodule:: mushroom_rl.environments.atari
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

Car on hill
~~~~~~~~~~~

.. automodule:: mushroom_rl.environments.car_on_hill
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

DeepMind Control Suite
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: mushroom_rl.environments.dm_control_env
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

Finite MDP
~~~~~~~~~~

.. automodule:: mushroom_rl.environments.finite_mdp
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

Grid World
~~~~~~~~~~

.. automodule:: mushroom_rl.environments.grid_world
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

Gym
~~~

.. automodule:: mushroom_rl.environments.gym_env
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

Habitat
~~~~~~~

.. autoclass:: mushroom_rl.environments.habitat_env.Habitat
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: mushroom_rl.environments.habitat_env.HabitatNavigationWrapper
    :members:

.. autoclass:: mushroom_rl.environments.habitat_env.HabitatRearrangeWrapper
    :members:


iGibson
~~~~~~~

.. autoclass:: mushroom_rl.environments.igibson_env.iGibson
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

Inverted pendulum
~~~~~~~~~~~~~~~~~

.. automodule:: mushroom_rl.environments.inverted_pendulum
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

Cart Pole
~~~~~~~~~

.. automodule:: mushroom_rl.environments.cart_pole
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

LQR
~~~

.. automodule:: mushroom_rl.environments.lqr
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

Minigrid
~~~~~~~~

.. automodule:: mushroom_rl.environments.minigrid_env
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:


Mujoco
~~~~~~

.. automodule:: mushroom_rl.environments.mujoco
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

Puddle World
~~~~~~~~~~~~

.. automodule:: mushroom_rl.environments.puddle_world
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

Pybullet
~~~~~~~~

.. automodule:: mushroom_rl.environments.pybullet
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

Air Hockey
^^^^^^^^^^

.. automodule:: mushroom_rl.environments.pybullet_envs.air_hockey.base
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.pybullet_envs.air_hockey.single
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.pybullet_envs.air_hockey.hit
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

.. automodule:: mushroom_rl.environments.pybullet_envs.air_hockey.defend
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

Segway
~~~~~~~~~~~~~

.. automodule:: mushroom_rl.environments.segway
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

Ship steering
~~~~~~~~~~~~~

.. automodule:: mushroom_rl.environments.ship_steering
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:
    
Generators
----------

Grid world
~~~~~~~~~~

.. automodule:: mushroom_rl.environments.generators.grid_world
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

Simple chain
~~~~~~~~~~~~

.. automodule:: mushroom_rl.environments.generators.simple_chain
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

Taxi
~~~~

.. automodule:: mushroom_rl.environments.generators.taxi
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:
