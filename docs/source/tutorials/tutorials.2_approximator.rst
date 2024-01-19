Create a regressor
=========================

MushroomRL offers a high-level interface to build function regressors. Indeed, it
transparently manages regressors for generic functions and Q-function regressors.
The user should not care about the low-level implementation of these regressors and
should only use the ``Regressor`` interface. This interface creates a Q-function regressor
or a ``GenericRegressor`` depending on whether the ``n_actions`` parameter is provided
to the constructor or not.

Usage of the ``Regressor`` interface
------------------------------------

When the action space of RL problems is finite and the adopted approach is value-based,
 we want to compute the Q-function of each action. In MushroomRL, this is possible using:

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
   :lines: 1-29

The following snippet, sets the output shape of the regressor to the number of
actions, creating a ``QRegressor``:

.. literalinclude:: code/approximator.py
   :lines: 30-32

If you prefer to use an ``ActionRegressor``, simply set the number of actions to (1,):

::

   approximator_params = dict(input_shape=(features.size,),
                              output_shape=(1,),
                              n_actions=mdp.info.action_space.n)

Then, the rest of the code fits the approximator and runs the evaluation rendering
the behaviour of the agent:

.. literalinclude:: code/approximator.py
   :lines: 33-

Generic regressor
-----------------
Whenever the ``n_actions`` parameter is not provided, the ``Regressor`` interface creates
a ``GenericRegressor``. This regressor can be used for general purposes and it is
more flexible to be used. It is commonly used in policy search algorithms.

Example
~~~~~~~

Create a dataset of points distributed on a line with random gaussian noise.

.. literalinclude:: code/generic_regressor.py
   :lines: 1-12

To fit the intercept, polynomial features of degree 1 are created by hand:

.. literalinclude:: code/generic_regressor.py
   :lines: 14

The regressor is then created and fit (note that ``n_actions`` is not provided):

.. literalinclude:: code/generic_regressor.py
   :lines: 16-20

Eventually, the approximated function of the regressor is plotted together with
the target points. Moreover, the weights and the gradient in point 5 of the linear approximator
are printed.

.. literalinclude:: code/generic_regressor.py
   :lines: 22-27
