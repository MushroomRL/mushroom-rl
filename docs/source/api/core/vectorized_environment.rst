Vectorized environment
======================

A ``VectorizedEnvironment`` steps a set of environment copies at once, returning one row per copy, and is driven by
``Core`` through the vectorized code path. ``MultiprocessEnvironment`` is the general-purpose implementation: it
builds the copies in separate processes, so that any single-environment implementation can be vectorized without
being rewritten. Environments backed by a simulator that is vectorized natively, e.g. the Isaac Sim ones, extend
``VectorizedEnvironment`` directly instead.

.. automodule:: mushroom_rl.core.vectorized_env
    :private-members:

.. automodule:: mushroom_rl.core.multiprocess_environment
    :private-members:
