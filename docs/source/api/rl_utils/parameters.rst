Parameters
==========

A parameter is a value used by an algorithm, such as a learning rate or an exploration coefficient. It can be a scalar,
a vector or, when a ``shape`` is given, a table holding one value per state or state-action tuple. Each parameter
declares an array ``backend`` (numpy by default) and, for torch, a ``device``.

Three families build on top of the base classes:

- *scheduled* parameters, whose value evolves with the number of updates;
- *variance-driven* parameters, whose value adapts to the observed variance of a target signal;
- *covariance* parameters, holding the covariance matrix of a Gaussian distribution, either fixed or built from a
  scheduled standard deviation.

.. automodule:: mushroom_rl.rl_utils.parameters.parameter
    :private-members:

Scheduled parameters
--------------------

.. automodule:: mushroom_rl.rl_utils.parameters.scheduled
    :private-members:

Variance-driven parameters
--------------------------

.. automodule:: mushroom_rl.rl_utils.parameters.variance
    :private-members:

Covariance parameters
---------------------

.. automodule:: mushroom_rl.rl_utils.parameters.covariance
    :private-members:
