Callbacks
=========

Callbacks are called by ``Core`` at every step or at every fit, and are the way to collect a quantity while an
experiment runs without modifying the agent. The collectors accumulate what they are given into a list that can be
read at the end of the run; ``DatasetMonitor`` instead feeds the live plotting windows.

.. automodule:: mushroom_rl.utils.callbacks.callback
    :private-members:

.. automodule:: mushroom_rl.utils.callbacks.collect_dataset

.. automodule:: mushroom_rl.utils.callbacks.collect_q

.. automodule:: mushroom_rl.utils.callbacks.collect_max_q

.. automodule:: mushroom_rl.utils.callbacks.collect_parameters

.. automodule:: mushroom_rl.utils.callbacks.dataset_monitor
