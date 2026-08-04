Prioritized replay memory
=========================

The variant drawing a transition with a probability that grows with its TD error, and correcting the resulting
bias with an importance-sampling weight returned together with the batch.

.. autoclass:: mushroom_rl.rl_utils.replay_memory.PrioritizedReplayMemory
    :private-members:
