Q-Approximator
==============

``QApproximator`` is the high-level interface for Q-function approximation, used in classical
(non-deep RL) algorithms with function approximators. This design allows to integrate many types of approximators,
including sklearn-style regressors seamlessly. Its constructor dispatches to one of three concrete implementations
depending on the arguments:

- ``QApproximatorSimple`` — single multi-output model (``output_shape[0] == n_actions``);
- ``QApproximatorAction`` — one independent model per action;
- ``QApproximatorEnsemble`` — ensemble of Q-approximators (``n_models > 1``).

.. automodule:: mushroom_rl.approximators.q_approximator
    :private-members:
