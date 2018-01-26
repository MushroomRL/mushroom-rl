How to create a regressor
=========================

Mushroom offers a high-level interface to build function regressors. Indeed, it
transparently manages regressors for generic functions and Q-function regressors.
The user should not care about the low-level implementation of these regressors and
should only use the ``Regressor`` interface. This interface creates a Q-function regressor
or a ``GenericRegressor`` depending on whether the ``n_actions`` parameter is provided
to the constructor or not.

Usage of the ``Regressor`` interface
------------------------------------

When the action space of RL problems is finite, the reasonable choice is to
have a different Q-function for each action. In Mushroom, this is possible using:

* a Q-function regressor with a different approximator for each action (``ActionRegressor``);
* a single Q-function regressor with a different output for each action (``QRegressor``).

The ``QRegressor`` is suggested when the number of discrete actions is high, due to
memory reasons.

The user can create create a ``QRegressor`` or an ``ActionRegressor``, setting
the ``output_shape`` parameter of the ``Regressor`` interface. If it is set to (1,),
an ``ActionRegressor`` is created; otherwise if it is set to the number of discrete actions,
a ``QRegressor`` is created.

Example
-------

Initially, the MDP, the policy and the features are created:

.. literalinclude:: code/approximator.py
   :lines: 1-32

The following snippet, sets the output shape of the regressor to the number of
actions, creating a ``QRegressor``:

.. literalinclude:: code/approximator.py
   :lines: 33-35

If you prefer to use an ``ActionRegressor``, simply set the number of actions to (1,):

::

   approximator_params = dict(input_shape=(features.size,),
                              output_shape=(1,),
                              n_actions=mdp.info.action_space.n)

Then, the rest of the code fits the approximator and runs the evaluation rendering
the behaviour of the agent:

.. literalinclude:: code/approximator.py
   :lines: 36-51

Generic regressor
-----------------
Whenever the ``n_actions`` parameter is not provided, the ``Regressor`` interface creates
a ``GenericRegressor``. This regressor can be used for general purposes and it is
more flexible to be used. It is commonly used in policy search algorithms.
