Dataset
=======

The ``Dataset`` stores the transitions collected while an agent interacts with an environment, keeping the
environment data and the agent (policy state) data in their respective backends. ``DatasetInfo`` carries the
static information used to build it, and ``VectorizedDataset`` handles data collected from parallel environments.

.. automodule:: mushroom_rl.core.dataset
    :private-members:
