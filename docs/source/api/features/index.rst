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

Every basis function, radial basis tensor and tiling accepts a ``dimensions`` argument,
to compute the features on a subspace of the input instead of the whole of it. The input
is always passed in full: it is the feature that selects the dimensions it needs, so
the same input can be given to features living in different subspaces. The arguments
describing the input, e.g. the ``low`` and ``high`` bounds of an environment
observation space, follow the same rule and describe the whole input, while the ones
describing the feature itself, e.g. the number of tiles or of centers, refer to the
selected dimensions only, in the same order they are declared in ``dimensions``. For
instance, the following builds tilings of 100 tiles on the first and third dimensions
of a three-dimensional observation space:

.. code-block:: python

    tilings = Tiles.generate(10, [10, 10], mdp.info.observation_space.low,
                             mdp.info.observation_space.high, dimensions=[0, 2])

.. autosummary::
   :nosignatures:

   ~mushroom_rl.features.features.Features

The ``Features`` interface and the documentation for every feature type can be found here:

.. toctree::
   :maxdepth: 1

   features
   basis
   tensors
   tiles
