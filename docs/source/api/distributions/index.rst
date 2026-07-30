Distributions
=============

A ``Distribution`` is a distribution over the parameter vector of a policy. It is the object the black-box
optimization algorithms learn: at the beginning of an episode they sample a parameter vector from it, run the
episode with the resulting policy, and update the distribution from the return that was obtained.

A distribution can be *contextual*, i.e. conditioned on a context vector built from the initial state and the
episode info, in which case sampling and updating are performed per context.

.. autosummary::
   :nosignatures:

   ~mushroom_rl.distributions.distribution.Distribution
   ~mushroom_rl.distributions.gaussian.GaussianDistribution
   ~mushroom_rl.distributions.gaussian.GaussianDiagonalDistribution
   ~mushroom_rl.distributions.gaussian.GaussianCholeskyDistribution

.. toctree::
   :maxdepth: 1

   distribution
   gaussian
   torch_distribution
