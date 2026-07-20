Reinforcement Learning utils
============================

Eligibility trace
-----------------

.. automodule:: mushroom_rl.rl_utils.eligibility_trace
    :members:
    :private-members:
    :undoc-members:
    :show-inheritance:

Optimizers
----------

.. automodule:: mushroom_rl.rl_utils.optimizers
    :members:
    :private-members:
    :undoc-members:
    :show-inheritance:

Parameters
----------

A parameter is a value used by an algorithm, such as a learning rate or an exploration coefficient. It can be
a scalar or, when a ``shape`` is given, a table holding one value per state or state-action tuple. Each
parameter declares an array ``backend`` (numpy by default); a non-scalar parameter converts the indexing
inputs into the format it requires, so a state from any backend can be used to index it directly. Two families build on top of
the base classes: *scheduled* parameters, whose value evolves with the number of updates, and *variance*
parameters, whose value adapts to the observed variance of a target signal.

.. automodule:: mushroom_rl.rl_utils.parameters.parameter
    :members:
    :private-members:
    :show-inheritance:

Scheduled parameters
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: mushroom_rl.rl_utils.parameters.scheduled
    :members:
    :private-members:
    :show-inheritance:

Variance parameters
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: mushroom_rl.rl_utils.parameters.variance
    :members:
    :private-members:
    :show-inheritance:


Preprocessors
-------------

.. automodule:: mushroom_rl.rl_utils.preprocessors
    :members:
    :private-members:
    :show-inheritance:

Replay memory
-------------

.. automodule:: mushroom_rl.rl_utils.replay_memory
    :members:
    :private-members:
    :show-inheritance:

Running Statistics
------------------

.. automodule:: mushroom_rl.rl_utils.running_stats
    :members:
    :private-members:
    :show-inheritance:

Spaces
------

.. automodule:: mushroom_rl.core.spaces
    :members:
    :show-inheritance:


Value Functions
---------------

.. automodule:: mushroom_rl.rl_utils.value_functions
    :members:
    :private-members:
    :show-inheritance:
