Value-Based
===========

Value based algorithms are algorithms learning a value function. As they do not learn an explicit control policy, but
the policy is derived from the value function, they are also called Critic-only.

TD
--
These are classical temporal difference algorithms for discrete actions. These algorithms cover both tabular methods and
function approximations.

.. automodule:: mushroom_rl.algorithms.value.td
    :members:
    :private-members:
    :show-inheritance:

Batch TD
--------

These are all batch TD methods, learning the Q-Function using a dataset of interaction with the environment.

.. automodule:: mushroom_rl.algorithms.value.batch_td
    :members:
    :private-members:
    :show-inheritance:

DQN
---

These methods are value-based Deep Reinforcement learning approaches. They are mostly variations of the DQN algorithm.

.. automodule:: mushroom_rl.algorithms.value.dqn
    :members:
    :private-members:
    :show-inheritance:
