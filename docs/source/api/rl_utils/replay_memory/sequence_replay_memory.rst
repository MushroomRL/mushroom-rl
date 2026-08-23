Sequence replay memory
======================

The variant returning contiguous sequences of transitions instead of independent ones, for the algorithms that
backpropagate through time and therefore need the temporal order preserved inside a sample.

.. autoclass:: mushroom_rl.rl_utils.replay_memory.SequenceReplayMemory
    :private-members:
