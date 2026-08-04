Solvers
=======

Exact solvers, computing the quantity a learning algorithm can only estimate. They are used to obtain the ground
truth an experiment is compared against: the optimal value function of a finite MDP, the optimal gain of an LQR
problem, and the optimal Q-function of the car-on-hill problem by exhaustive search.

.. autosummary::
   :nosignatures:

   ~mushroom_rl.solvers.dynamic_programming.value_iteration
   ~mushroom_rl.solvers.dynamic_programming.policy_iteration
   ~mushroom_rl.solvers.car_on_hill.solve_car_on_hill
   ~mushroom_rl.solvers.lqr.compute_lqr_feedback_gain

.. toctree::
   :maxdepth: 1

   dynamic_programming
   car_on_hill
   lqr
