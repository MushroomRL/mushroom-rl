Torch approximator
==================

Wraps a PyTorch network into the approximator interface, taking care of the optimizer, the loss and the fitting
loop, and exposing the weights and the gradient in the flat form the policy-search algorithms expect.

.. automodule:: mushroom_rl.approximators.parametric.torch_approximator
    :private-members:
