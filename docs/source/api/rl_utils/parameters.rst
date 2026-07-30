Parameters
==========

A parameter is a value used by an algorithm, such as a learning rate or an exploration coefficient. It can be a scalar
or, when a ``shape`` is given, a table holding one value per state or state-action tuple. Each parameter declares an
array ``backend`` (numpy by default); a non-scalar parameter converts the indexing inputs into the format it requires,
so a state from any backend can be used to index it directly.

Two families build on top of the base classes:
- *scheduled* parameters, whose value evolves with the number of updates;
- *variance* parameters, whose value adapts to the observed variance of a target signal.

.. automodule:: mushroom_rl.rl_utils.parameters.parameter
    :private-members:

Scheduled parameters
--------------------

.. automodule:: mushroom_rl.rl_utils.parameters.scheduled
    :private-members:

Variance parameters
-------------------

.. automodule:: mushroom_rl.rl_utils.parameters.variance
    :private-members:
