Stateful Torch policy
=====================

Torch policies carrying a latent state across the steps of an episode, e.g. the hidden state of a recurrent network.
The state is returned together with the action and stored in the dataset, so that it can be replayed when the algorithm
fits on the collected sequences.

.. automodule:: mushroom_rl.policy.stateful_torch_policy
    :private-members:
