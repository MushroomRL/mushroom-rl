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

.. autosummary::
   :nosignatures:

   ~mushroom_rl.approximators.approximator.Approximator
   ~mushroom_rl.approximators.q_approximator.QApproximator
   ~mushroom_rl.approximators.table.Table
   ~mushroom_rl.approximators.parametric.linear.LinearApproximator
   ~mushroom_rl.approximators.parametric.cmac.CMAC
   ~mushroom_rl.approximators.parametric.torch_approximator.TorchApproximator
   ~mushroom_rl.approximators.parametric.recurrent_torch_approximator.RecurrentTorchApproximator

.. toctree::
   :maxdepth: 1

   approximator
   q_approximator
   table
   parametric/index
