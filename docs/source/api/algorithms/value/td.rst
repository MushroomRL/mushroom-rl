Temporal difference
===================

These are classical temporal difference algorithms for discrete actions. These algorithms cover both tabular methods and
function approximations.

They update the Q-function from a single transition as soon as it is observed, so they are fitted one step at a
time. The ``TD`` base implements the loop and leaves the update rule to the subclass.

.. autoclass:: mushroom_rl.algorithms.value.td.TD
    :private-members:

.. automodule:: mushroom_rl.algorithms.value.td
    :private-members:
