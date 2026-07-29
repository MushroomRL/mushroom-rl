Agent-Environment Interface
===========================

The three basic interface of mushroom_rl are the Agent, the Environment and the Core interface.

- The ``Agent`` is the basic interface for any Reinforcement Learning algorithm.
- The ``Environment`` is the basic interface for every problem/task that the agent should solve.
- The ``Core`` is a class used to control the interaction between an agent and an environment.

We provide the logging functionality with the ``Logger`` class. Finally, the ``MushroomObject`` interface implements
serialization of MushroomRL data on the disk (load/save functionality), forwards a logger down the object tree, and
names its objects through the ``name`` and ``full_name`` methods.


Agent
-----

MushroomRL provides the implementations of several algorithms belonging to all
categories of RL:

- value-based;
- policy-search;
- actor-critic.

One can easily implement customized algorithms following the structure of the
already available ones, by extending the following interface:

.. automodule:: mushroom_rl.core.agent
    :members:
    :private-members:
    :show-inheritance:

Environment
-----------

MushroomRL provides several implementation of well known benchmarks with both
continuous and discrete action spaces.

To implement a new environment, it is mandatory to use the following interface:

Every environment is identified by the ``name`` class method, which returns the class name and is the key under
which ``register`` stores the environment and ``make`` looks it up. Environments standing for a family of tasks
override the ``full_name`` instance method to append the specific task they were built with, using the same ``'.'``
separator that ``make`` accepts, so that the returned string rebuilds the environment. The ``Logger`` uses these
names to label an experiment.

.. automodule:: mushroom_rl.core.environment
    :members:
    :private-members:
    :show-inheritance:


Core
----

.. automodule:: mushroom_rl.core.core
    :members:
    :private-members:
    :show-inheritance:

Dataset
-------

The ``Dataset`` stores the transitions collected while an agent interacts with an environment, keeping the
environment data and the agent (policy state) data in their respective backends. ``DatasetInfo`` carries the
static information used to build it, and ``VectorizedDataset`` handles data collected from parallel environments.

.. automodule:: mushroom_rl.core.dataset
    :members:
    :private-members:
    :show-inheritance:

History manager
---------------

.. automodule:: mushroom_rl.core.history_manager
    :members:
    :private-members:
    :show-inheritance:

Array Backend
-------------

.. automodule:: mushroom_rl.core.array_backend
    :members:
    :private-members:
    :show-inheritance:

Serialization
-------------

.. automodule:: mushroom_rl.core.mushroom_object
    :members:
    :private-members:
    :show-inheritance:

Logger
------

.. automodule:: mushroom_rl.core.logger
    :members:
    :private-members:
    :show-inheritance:
