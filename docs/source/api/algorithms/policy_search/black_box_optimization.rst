Black-box optimization
======================

Black-box optimization methods explore in parameter space: a ``Distribution`` draws a policy parameter vector at
the beginning of each episode, and is updated from the returns the sampled parameters achieved. The policy is
optimized as a black box, and the algorithm never sees the individual transitions — only the return and
the parameters that produced it. When the distribution is contextual, a ``ContextBuilder`` maps the initial state
and the episode info to the context it is conditioned on.

.. automodule:: mushroom_rl.algorithms.policy_search.black_box_optimization
    :private-members:
