Deep actor-critic
=================

.. module:: mushroom_rl.algorithms.actor_critic.deep_actor_critic

The deep actor-critic algorithms represent both actor and critic with PyTorch networks. They split in two groups,
which is also how the base classes are organized:

- **off-policy** methods keep a replay memory and target networks, and differentiate through the sampled action
  with the reparametrization trick; they build the policy themselves, so they act on continuous actions;
- **on-policy** methods fit on the batch of samples they just collected, and take the policy from the caller, so
  the action space is the one their policy supports.

.. autosummary::
   :nosignatures:

   DeepAC
   OnPolicyDeepAC
   DDPG
   TD3
   SAC
   A2C
   TRPO
   PPO
   PPO_BPTT
   RudinPPO

.. toctree::
   :maxdepth: 1

   deep_actor_critic
   off_policy
   on_policy
