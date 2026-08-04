Finite MDPs
===========

Some environments, e.g. grid worlds, are finite Markov Decision Processes: they extend the ``FiniteMDP`` class,
exposing the transition probability matrix, the reward matrix and the initial state distribution, so that they can
also be solved with the dynamic programming solvers.

An absorbing state is represented as a self-loop with zero reward, so that the dynamic programming solvers converge
to the same value function the sampling algorithms estimate.

.. autosummary::
   :nosignatures:

   ~mushroom_rl.environments.finite_mdp.FiniteMDP
   ~mushroom_rl.environments.grid_world.GridWorld
   ~mushroom_rl.environments.grid_world_van_hasselt.GridWorldVanHasselt
   ~mushroom_rl.environments.simple_chain.SimpleChain
   ~mushroom_rl.environments.taxi.Taxi

.. toctree::
   :maxdepth: 1

   finite_mdp
   grid_world
   simple_chain
   taxi
