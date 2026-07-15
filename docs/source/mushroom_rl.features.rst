Features
========

The features in MushroomRL are 1-D arrays computed applying a specified function
to a raw input, e.g. polynomial features of the state of an MDP.
MushroomRL supports four types of features:

* basis functions;
* tensor basis functions;
* tiles;
* functional mappings.

The basis functions are a plain numpy implementation: simple to read, and a good
starting point to write your own basis. The tensor basis functions are the PyTorch
counterpart: the implementation is harder to follow, but they are faster to compute,
as they can exploit parallel computing, e.g. GPU-acceleration and multi-core systems,
and they can be turned into a differentiable ``torch`` module with
``to_torch_module``, to be embedded inside a network. The tiles discretize the input
space, returning the one-hot encoding of the tile the input falls into. A functional
mapping simply applies a given function to the raw input.

All the types of features are exposed by a single ``Features`` class, that builds
the one requested by the user from the ``feature_list`` passed at construction, or
from ``n_outputs`` and ``function`` for a functional mapping. Different types of
features cannot be mixed in the same ``feature_list``. Whatever the type, the backend
of the computed features can be selected with the ``backend`` argument.

.. automodule:: mushroom_rl.features.features
    :members:
    :private-members:
    :show-inheritance:

The documentation for every feature type can be found here:

.. toctree::

    features/mushroom_rl.features.basis
    features/mushroom_rl.features.tensors
    features/mushroom_rl.features.tiles
