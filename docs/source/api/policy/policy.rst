Policy interface
================

The interface every policy implements, and the two mixins that extend it with trainable weights and with the
gradient of the log-probability. ``StatefulPolicy`` adds the latent internal state carried across the steps of an
episode and stored in the dataset.

.. automodule:: mushroom_rl.policy.policy
    :private-members:
