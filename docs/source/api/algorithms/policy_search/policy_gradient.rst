Policy gradient
===============

Policy-gradient methods explore in action space and estimate the gradient of the expected return with respect to
the policy parameters. They require a policy exposing the gradient of its log-probability, and more than one
episode per fit, since the gradient is averaged over the episode returns.

.. automodule:: mushroom_rl.algorithms.policy_search.policy_gradient
    :private-members:
