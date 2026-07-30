Deep actor-critic interfaces
============================

The two bases the deep actor-critic algorithms extend. ``DeepAC`` covers the off-policy, reparametrized ones,
holding the optimizer over the actor parameters; ``OnPolicyDeepAC`` covers the on-policy ones, adding the
diagnostics they log at every fit.

.. autoclass:: mushroom_rl.algorithms.actor_critic.deep_actor_critic.DeepAC
    :private-members:

.. autoclass:: mushroom_rl.algorithms.actor_critic.deep_actor_critic.OnPolicyDeepAC
    :private-members:
