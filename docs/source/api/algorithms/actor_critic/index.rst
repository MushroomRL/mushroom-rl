Actor-critic
============

Actor-critic methods learn a policy (the *actor*) and a value function (the *critic*) at the same time, using the
critic to build a lower-variance estimate of the gradient the actor is updated with.

The classical methods are the linear, feature-based formulations: they act on continuous actions and update at
every step, using an eligibility trace on both actor and critic. The deep ones represent actor and critic with
PyTorch networks.

.. autosummary::
   :nosignatures:

   ~mushroom_rl.algorithms.actor_critic.classic_actor_critic.COPDAC_Q
   ~mushroom_rl.algorithms.actor_critic.classic_actor_critic.StochasticAC
   ~mushroom_rl.algorithms.actor_critic.deep_actor_critic.DeepAC
   ~mushroom_rl.algorithms.actor_critic.deep_actor_critic.OnPolicyDeepAC

.. toctree::
   :maxdepth: 1

   classic
   deep/index
