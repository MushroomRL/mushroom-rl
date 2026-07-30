Replay memory
=============

.. module:: mushroom_rl.rl_utils.replay_memory

The buffers the off-policy algorithms sample their batches from. A replay memory stores the transitions as they are
collected and returns an independent batch on request, which is what breaks the correlation between consecutive samples
that would otherwise destabilize the fit.

The three implementations differ in what a sample is and how it is drawn: uniformly over single transitions, over
sequences for the recurrent algorithms, or with a probability proportional to the TD error.

.. autosummary::
   :nosignatures:

   ReplayMemory
   SequenceReplayMemory
   PrioritizedReplayMemory

.. toctree::
   :maxdepth: 1

   replay_memory
   sequence_replay_memory
   prioritized_replay_memory
