Classical actor-critic
======================

The classical, non-deep actor-critic methods. They act on continuous actions and update at every step from the
transition just observed. ``StochasticAC`` and ``StochasticAC_AVG`` keep an eligibility trace for both the actor and
the critic; ``COPDAC_Q`` uses none.

.. automodule:: mushroom_rl.algorithms.actor_critic.classic_actor_critic
    :private-members:
