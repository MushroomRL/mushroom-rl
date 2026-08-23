Replay memory
=============

The uniform replay memory: it stores the raw transitions in a circular buffer and returns a batch drawn uniformly
among them, reducing the n-step return at sampling time so that the same buffer serves any n.

.. autoclass:: mushroom_rl.rl_utils.replay_memory.ReplayMemory
    :private-members:
