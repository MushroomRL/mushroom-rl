Preprocessors
=============

Transformations applied to the observation before it reaches the agent. They are stateful — a standardization
preprocessor keeps the running mean and variance of what it has seen — so they are serialized with the agent and
must be given to ``Core`` as core preprocessors, or to the agent as agent preprocessors, depending on whether the
stored dataset should hold the raw or the transformed observation.

.. automodule:: mushroom_rl.rl_utils.preprocessors
    :private-members:
