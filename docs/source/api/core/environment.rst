Environment
===========

MushroomRL provides several implementation of well known benchmarks with both
continuous and discrete action spaces.

To implement a new environment, it is mandatory to use the following interface:

Every environment is identified by the ``name`` class method, which returns the class name and is the key under
which ``register`` stores the environment and ``make`` looks it up. Environments standing for a family of tasks
override the ``full_name`` instance method to append the specific task they were built with, using the same ``'.'``
separator that ``make`` accepts, so that the returned string rebuilds the environment. The ``Logger`` uses these
names to label an experiment.

.. automodule:: mushroom_rl.core.environment
    :private-members:
