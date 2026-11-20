Deep actor-critic
=================

.. module:: mushroom_rl.algorithms.actor_critic.deep_actor_critic

The deep actor-critic algorithms represent both actor and critic with PyTorch networks. They split in two groups,
which is also how their base classes are organized:

- **off-policy** methods, :class:`DDPG`, :class:`TD3` and :class:`SAC`, learn from a replay memory and keep target
  networks. They update the actor by backpropagating the critic's Q-value through the action the actor produces: the
  deterministic action for DDPG and TD3, a reparametrized sample for SAC. They build the policy themselves, so they
  act on continuous actions;
- **on-policy** methods fit on the batch of samples they just collected, with a state-value critic estimating the
  advantage of the actions taken, and take the policy from the caller, so the action space is the one their policy
  supports. :class:`A2C` estimates the advantage from Monte Carlo returns; :class:`TRPO` and :class:`PPO` use
  generalized advantage estimation, and keep every policy update small, TRPO within a trust region and PPO by
  clipping the probability ratio. :class:`PPO_BPTT` extends PPO to recurrent policies, and :class:`RudinPPO` adds
  gradient clipping and a learning rate adapted to the KL divergence.

The base classes these algorithms extend, to build a new one on, are listed in the following table and documented in
:doc:`deep_actor_critic`.

.. autosummary::
   :nosignatures:

   DeepAC
   OnPolicyDeepAC
   OffPolicyDeepAC

.. toctree::
   :maxdepth: 1

   deep_actor_critic
   off_policy
   on_policy
