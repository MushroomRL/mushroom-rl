On-policy methods
=================

On-policy deep actor-critic algorithms. They fit on the batch of samples just collected and then discard it, and
they take the policy as a constructor argument, so the action space is whatever that policy supports.

.. autoclass:: mushroom_rl.algorithms.actor_critic.deep_actor_critic.A2C

.. autoclass:: mushroom_rl.algorithms.actor_critic.deep_actor_critic.TRPO

.. autoclass:: mushroom_rl.algorithms.actor_critic.deep_actor_critic.PPO

.. autoclass:: mushroom_rl.algorithms.actor_critic.deep_actor_critic.PPO_BPTT

.. autoclass:: mushroom_rl.algorithms.actor_critic.deep_actor_critic.RudinPPO
