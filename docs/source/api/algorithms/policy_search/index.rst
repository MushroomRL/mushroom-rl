Policy search
=============

Policy-search methods optimize the policy directly, without learning a value function to derive it from. They are
episodic: ``fit`` computes a return per episode, so it needs complete episodes and ``Core`` must be run with
``n_episodes_per_fit``.

The two families differ in where the exploration happens. **Policy-gradient** methods explore in action space,
perturbing the action at every step, and estimate the gradient of the expected return with respect to the policy
parameters from the log-probability of the actions taken. **Black-box optimization** methods explore in parameter
space instead: a ``Distribution`` samples a parameter vector once per episode, the episode is run with the
resulting deterministic policy, and the distribution is updated from the returns.

.. autosummary::
   :nosignatures:

   ~mushroom_rl.algorithms.policy_search.policy_gradient.PolicyGradient
   ~mushroom_rl.algorithms.policy_search.black_box_optimization.BlackBoxOptimization

.. toctree::
   :maxdepth: 1

   policy_gradient
   black_box_optimization
