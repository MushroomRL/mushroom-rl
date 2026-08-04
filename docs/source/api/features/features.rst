Features interface
==================

``Features`` is the single entry point to every feature type: it builds the one requested from the
``feature_list`` passed at construction, or a functional mapping from ``n_outputs`` and ``function``, and exposes
it through a uniform call interface whatever the underlying implementation is.

.. automodule:: mushroom_rl.features.features
    :private-members:
