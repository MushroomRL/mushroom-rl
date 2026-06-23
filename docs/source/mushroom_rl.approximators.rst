Approximators
=============

MushroomRL provides a hierarchy of approximator classes for both tabular and
function-approximation settings.

The base class ``Approximator`` dispatches to an ``Ensemble`` of models when
``n_models > 1`` is passed to the constructor. This means any approximator
subclass (``Table``, ``LinearApproximator``, …) can be
turned into an ensemble simply by passing ``n_models``:

.. code-block:: python

    # single model
    q = Table(shape=(10, 4))

    # ensemble of 5 tables — returns an Ensemble instance transparently
    q = Table(n_models=5, shape=(10, 4))


Approximator
------------

.. automodule:: mushroom_rl.approximators.approximator
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:


Q-Approximator
--------------

``QApproximator`` is the high-level interface for Q-function approximation, used in classical
(non-deep RL) algorithms with function approximators. This design allows to integrate many types of approximators, 
including sklearn-style regressors seamlessly. Its constructor dispatches to one of three concrete implementations 
depending on the arguments:

- ``QApproximatorSimple`` — single multi-output model (``output_shape[0] == n_actions``);
- ``QApproximatorAction`` — one independent model per action;
- ``QApproximatorEnsemble`` — ensemble of Q-approximators (``n_models > 1``).


.. automodule:: mushroom_rl.approximators.q_approximator
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

Tabular
-------

The simplest approximation type in RL is the table. The Tabular approximator can be used in settings where both state
and action are discrete, or can be discretized in a simple way.

.. automodule:: mushroom_rl.approximators.table
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:



Parametric
----------

The parametric approximators are the most common ones, and allow to learn a function by learning its parameter vector.
This approximators are often also differentiable.
Mushroom implements many common parametric approximators, and allows the user to set the parameters and compute the
gradient.


Linear
~~~~~~

.. automodule:: mushroom_rl.approximators.parametric.linear
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:
    
CMAC
~~~~

.. automodule:: mushroom_rl.approximators.parametric.cmac
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

Torch Approximator
~~~~~~~~~~~~~~~~~~

.. automodule:: mushroom_rl.approximators.parametric.torch_approximator
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

Networks
~~~~~~~~

Pre-built PyTorch network architectures for use with ``TorchApproximator``.

.. automodule:: mushroom_rl.approximators.parametric.networks.atari_network
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

.. automodule:: mushroom_rl.approximators.parametric.networks.actor_network
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

.. automodule:: mushroom_rl.approximators.parametric.networks.critic_network
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

.. automodule:: mushroom_rl.approximators.parametric.networks.q_network
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

.. automodule:: mushroom_rl.approximators.parametric.networks.linear_network
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

.. automodule:: mushroom_rl.approximators.parametric.networks.dueling_network
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

.. automodule:: mushroom_rl.approximators.parametric.networks.noisy_network
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

.. automodule:: mushroom_rl.approximators.parametric.networks.categorical_network
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

.. automodule:: mushroom_rl.approximators.parametric.networks.quantile_network
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

.. automodule:: mushroom_rl.approximators.parametric.networks.rainbow_network
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

.. automodule:: mushroom_rl.approximators.parametric.networks.recurrent_network
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:
