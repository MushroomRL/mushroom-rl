History manager
===============

The ``HistoryManager`` assembles the *context* a policy sees when it depends on more than the current observation,
e.g. a window of stacked observations or of previous actions. Being a deterministic function of the observed
trajectory, the context is reconstructed from the stored transitions rather than saved in the dataset, so the same
manager serves both the online loop and the offline replay of a batch.

.. automodule:: mushroom_rl.core.history_manager
    :private-members:
