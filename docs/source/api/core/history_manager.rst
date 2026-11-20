History manager
===============

The ``HistoryManager`` assembles the *context* a policy sees when it depends on more than the current observation,
e.g. a window of stacked observations or of previous actions. Being a deterministic function of the observed
trajectory, the context is reconstructed from the stored transitions rather than saved in the dataset, so the same
manager serves both the online loop and the offline replay of a batch.

.. automodule:: mushroom_rl.core.history_manager
    :private-members:

Asymmetric inputs
-----------------

The ``AsymmetricHistoryManager`` builds two inputs from different entries of the preprocessed observation, each with
its own history length: the policy input, online and when parsing a dataset, and the critic input, when parsing a
dataset only, into the ``critic_state`` and ``critic_next_state`` of the returned batch.

.. automodule:: mushroom_rl.core.asymmetric_history_manager
