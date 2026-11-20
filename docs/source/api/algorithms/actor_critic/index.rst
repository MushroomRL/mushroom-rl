Actor-critic
============

Actor-critic methods learn a policy (the *actor*) and a value function (the *critic*) at the same time, using the
critic to build a lower-variance estimate of the gradient the actor is updated with.

The **classical** methods are linear, feature-based formulations: they act on continuous actions and update at every
step. :class:`~mushroom_rl.algorithms.actor_critic.classic_actor_critic.StochasticAC` follows the gradient of a
stochastic policy, with an eligibility trace on both actor and critic, and
:class:`~mushroom_rl.algorithms.actor_critic.classic_actor_critic.StochasticAC_AVG` is its average-reward variant.
:class:`~mushroom_rl.algorithms.actor_critic.classic_actor_critic.COPDAC_Q` learns a deterministic policy off-policy,
acting with a separate behaviour policy, from a critic built on features compatible with the deterministic policy.

The **deep** methods represent actor and critic with PyTorch networks. The off-policy ones learn from a replay
memory and update the actor through the critic's Q-value; the on-policy ones fit on the samples just collected, with
a state-value critic estimating the advantage of the actions taken. :doc:`deep/index` describes each of them.

.. autosummary::
   :nosignatures:

   ~mushroom_rl.algorithms.actor_critic.classic_actor_critic.ClassicAC
   ~mushroom_rl.algorithms.actor_critic.deep_actor_critic.DeepAC

.. toctree::
   :maxdepth: 1

   classic
   deep/index
