Reinforcement learning utils
============================

This module contains the building blocks the algorithms are assembled from.
They are collected here rather than inside the algorithms because most of them are shared by several families — a
``Parameter`` schedules a learning rate as readily as an exploration coefficient, and the same advantage estimator
serves every on-policy actor-critic.

.. autosummary::
   :nosignatures:

   ~mushroom_rl.rl_utils.parameters.parameter.Parameter
   ~mushroom_rl.rl_utils.optimizers.Optimizer
   ~mushroom_rl.rl_utils.preprocessors.StandardizationPreprocessor
   ~mushroom_rl.rl_utils.replay_memory.ReplayMemory
   ~mushroom_rl.rl_utils.eligibility_trace.EligibilityTrace
   ~mushroom_rl.rl_utils.running_stats.RunningStandardization

.. toctree::
   :maxdepth: 1

   parameters
   optimizers
   preprocessors
   replay_memory/index
   eligibility_trace
   running_stats
   value_functions
