Policy interface
================

The interface every policy implements, and the mixins that extend it with trainable weights, with the gradient of
the log-probability and with the location of the action distribution. ``StatefulPolicy`` adds the latent internal
state carried across the steps of an episode and stored in the dataset.

.. automodule:: mushroom_rl.policy.policy
    :private-members:
