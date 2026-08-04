Batch temporal difference
=========================

These are all batch TD methods, learning the Q-Function using a dataset of interaction with the environment.

Instead of a single update per transition, they refit the whole Q-function on the dataset at every iteration, which
makes them a natural fit for the sklearn-style approximators exposed through ``QApproximator``.

.. automodule:: mushroom_rl.algorithms.value.batch_td
    :private-members:
