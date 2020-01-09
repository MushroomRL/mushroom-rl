Agent-Environment Interface
===========================

The three basic interface of mushroom_rl are the Agent, the Environment and the Core interface.

- The ``Agent`` is the basic interface for any Reinforcement Learning algorithm.
- The ``Environment`` is the basic interface for every problem/task that the agent should solve.
- The ``Core`` is a class used to control the interaction between an agent and an environment.


Agent
-----

MushroomRL provides the implementations of several algorithms belonging to all
categories of RL:

- value-based;
- policy-search;
- actor-critic.

One can easily implement customized algorithms following the structure of the
already available ones, by extending the following interface:

.. automodule:: mushroom_rl.algorithms.agent
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

Environment
-----------

MushroomRL provides several implementation of well known benchmarks with both
continuous and discrete action spaces.

To implement a new environment, it is mandatory to use the following interface:

.. automodule:: mushroom_rl.environments.environment
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:


Core
----

.. automodule:: mushroom_rl.core.core
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:
