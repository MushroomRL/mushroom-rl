Deep actor-critic interfaces
============================

Base classes of the deep actor-critic algorithms. ``DeepAC`` is the base class of every deep actor-critic algorithm
and holds the optional optimizer of the actor parameters; ``OnPolicyDeepAC`` adds the dataset parsing and the
diagnostics logged at every fit by the on-policy algorithms; ``OffPolicyDeepAC`` adds the initialization and the soft
update of the target networks used by the off-policy algorithms.

.. autoclass:: mushroom_rl.algorithms.actor_critic.deep_actor_critic.DeepAC
    :private-members:

.. autoclass:: mushroom_rl.algorithms.actor_critic.deep_actor_critic.OnPolicyDeepAC
    :private-members:

.. autoclass:: mushroom_rl.algorithms.actor_critic.deep_actor_critic.OffPolicyDeepAC
    :private-members:
