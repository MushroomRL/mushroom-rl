How to create a regressor
=========================

Mushroom offers a high-level interface to build function regressors. Indeed, it
transparently manages regressors for generic functions and Q-function regressors.
The user should not care about the low-level implementation of these regressors and
should only use the ``Regressor`` interface. This interface creates a Q-regressor
or a generic regressor depending on whether the number of actions is provided
or not to the constructor.

Q-function regressor
--------------------

When the number of actions is provided, it is reasonable that a Q-regressor should be created.
There are two types of Q-regressors:

* a Q-regressor with a different regressor for each action (``ActionRegressor``);
* a single Q-regressor with a different output for each action (``QRegressor``).

The choice of whether to create a ``QRegressor`` or an ``ActionRegressor``, is done
checking whether the ``output_shape`` parameter is equal to the number of actions or
not:

::

    approximator_params = dict(input_shape=mdp.info.observation_space.shape,
                               output_shape=(mdp.info.action_space.n,),
                               n_actions=mdp.info.action_space.n,
                               n_estimators=50,
                               min_samples_split=5,
                               min_samples_leaf=2)

In this example, the number of actions is equal to the shape of the output,
therefore a ``QRegressor`` is created.

::

    approximator_params = dict(input_shape=mdp.info.observation_space.shape,
                               n_actions=mdp.info.action_space.n,
                               n_estimators=50,
                               min_samples_split=5,
                               min_samples_leaf=2)

The ``output_shape`` is not provided, so the default :code:`output_shape=(1,)`
is used. This way, an ``ActionRegressor`` is created.

Generic regressor
-----------------
Whenever the number of actions is not provided, the Regressor interface creates
a ``GenericRegressor``. This regressor can be used for general purposes and it is
more flexible to be used.
