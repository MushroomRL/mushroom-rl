Parametric approximators
========================

The parametric approximators are the most common ones, and allow to learn a function by learning its parameter vector.
This approximators are often also differentiable.
Mushroom implements many common parametric approximators, and allows the user to set the parameters and compute the
gradient.

.. autosummary::
   :nosignatures:

   ~mushroom_rl.approximators.parametric.linear.LinearApproximator
   ~mushroom_rl.approximators.parametric.cmac.CMAC
   ~mushroom_rl.approximators.parametric.torch_approximator.TorchApproximator
   ~mushroom_rl.approximators.parametric.recurrent_torch_approximator.RecurrentTorchApproximator

.. toctree::
   :maxdepth: 1

   linear
   cmac
   torch_approximator
   recurrent_torch_approximator
   networks/index
