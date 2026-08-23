Value-based
===========

Value based algorithms are algorithms learning a value function. As they do not learn an explicit control policy, but
the policy is derived from the value function, they are also called Critic-only.

All of them act on a discrete action space: the Q-function is evaluated on every action to take the greedy one,
which is what the tabular update and the target of a DQN both do. What separates the three families below is where
the samples come from — one transition at a time, a fixed dataset, or a replay memory.

.. autosummary::
   :nosignatures:

   ~mushroom_rl.algorithms.value.td.TD
   ~mushroom_rl.algorithms.value.batch_td.BatchTD
   ~mushroom_rl.algorithms.value.dqn.AbstractDQN

.. toctree::
   :maxdepth: 1

   td
   batch_td
   dqn
