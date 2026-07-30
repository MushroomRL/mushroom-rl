Off-policy methods
==================

Off-policy deep actor-critic algorithms. They learn from a replay memory, keep target networks for the critic, and
build their own policy on top of the actor approximator, so they act on continuous actions.

.. autoclass:: mushroom_rl.algorithms.actor_critic.deep_actor_critic.DDPG

.. autoclass:: mushroom_rl.algorithms.actor_critic.deep_actor_critic.TD3

.. autoclass:: mushroom_rl.algorithms.actor_critic.deep_actor_critic.SAC
