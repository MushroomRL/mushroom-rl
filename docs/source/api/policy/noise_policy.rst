Noise policy
============

Policies adding exploration noise on top of a deterministic actor. The Ornstein-Uhlenbeck policy is stateful, as
its noise is correlated in time and must be carried across the steps of an episode; the clipped Gaussian one draws
independent noise and clips the result to the action range.

.. automodule:: mushroom_rl.policy.noise_policy
    :private-members:
