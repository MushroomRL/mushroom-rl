How to use approximators
========================

MushroomRL provides an interface for approximator classes to perform function approximation.
The root class is ``Approximator``; every concrete subclass (``Table``, ``LinearApproximator``,
``TorchApproximator``, …) automatically dispatches to an ``Ensemble`` when
``n_models > 1`` is passed to the constructor, allowing ensemble methods to
be built without any extra glue code:

.. code-block:: python

    # single table
    q = Table(shape=(10, 4))

    # ensemble of 5 tables — returns an Ensemble transparently
    q = Table(n_models=5, shape=(10, 4))

Function approximation
----------------------

An approximator is a class representing any type of function approximation (parametric, non-parametric, tabular).
It exposes a ``fit`` / ``predict`` interface, and  in case of parametric functions, provides weight and gradient access.

The example below fits a ``LinearApproximator`` to points sampled from a
line with Gaussian noise. Polynomial features of degree 1 are built by hand
so that the model can learn both slope and intercept:

.. literalinclude:: code/approximator.py
   :lines: 1-16

After fitting, the weights, the gradient at a specific input, and a plot of
the approximated function can be obtained:

.. literalinclude:: code/approximator.py
   :lines: 18-

Features and input shape
------------------------

Instead of building the features by hand, a ``Features`` object can be given to the approximator
through the ``phi`` parameter. The approximator then featurizes its input internally, so ``fit``
and ``predict`` are called with the raw input:

.. code-block:: python

    phi = Features(PolynomialBasis.generate(1, 1))
    approximator = LinearApproximator(input_shape=(1,), output_shape=(1,), phi=phi)
    approximator.fit(x, y)

``input_shape`` always describes the input of the model, i.e. what is passed to ``fit`` and
``predict``. When ``phi`` is given, the number of weights is determined by the size of the
features, not by ``input_shape``: the approximator above holds two weights, one per polynomial
feature, even though its input is one-dimensional. ``input_shape`` must be a tuple, as an
approximator takes a single input.

Q-function approximation
------------------------

For classical RL algorithms with discrete action spaces, MushroomRL provides
``QApproximator`` — a unified interface that selects the appropriate concrete
implementation based on the constructor arguments:

* ``n_models > 1``: ``QApproximatorEnsemble`` — ensemble of independent
  Q-approximators;
* ``output_shape[0] != n_actions``: ``QApproximatorAction`` — one independent
  model per action;
* ``output_shape[0] == n_actions``: ``QApproximatorSimple`` — a single
  multi-output model with one output per action.

Algorithms that accept a parametric ``approximator`` class (e.g. ``SARSALambdaContinuous``,
``TrueOnlineSARSALambda``, ``FQI``) pass it through ``QApproximator`` internally, so the
same algorithm code handles all three cases transparently.

``QApproximatorSimple`` is preferred when the number of actions is large, since
a single model stores all Q-values jointly. ``QApproximatorAction`` trains a
separate model per action and is useful when per-action function complexity
differs.

Example
~~~~~~~

The following example trains a SARSA(λ) agent on the MountainCar environment
using tile-coded features and a ``LinearApproximator``.

First, the MDP, the policy and the features are set up:

.. literalinclude:: code/q_approximator.py
   :lines: 1-24

Setting ``output_shape`` to the number of actions creates a ``QApproximatorSimple``
inside the algorithm:

.. literalinclude:: code/q_approximator.py
   :lines: 26-35

To use a ``QApproximatorAction`` instead — one independent model per action —
simply set ``output_shape`` to ``(1,)``:

.. code-block:: python

   approximator_params = dict(input_shape=mdp.info.observation_space.shape,
                              output_shape=(1,),
                              n_actions=mdp.info.action_space.n,
                              phi=features)

The rest creates the training loop and runs training and evaluation:

.. literalinclude:: code/q_approximator.py
   :lines: 37-