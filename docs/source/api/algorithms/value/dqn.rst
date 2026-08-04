DQN
===

These methods are value-based Deep Reinforcement learning approaches. They are mostly variations of the DQN algorithm.

They all share the ``AbstractDQN`` skeleton — a replay memory, an online and a target approximator, and a periodic
target update — and differ in how the target value ``_next_q`` is computed and in the network architecture they
expect.

.. automodule:: mushroom_rl.algorithms.value.dqn
    :private-members:
