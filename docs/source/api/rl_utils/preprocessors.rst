Preprocessors
=============

Transformations applied to the observation before it reaches the agent. They are stateful — a standardization
preprocessor keeps the running mean and variance of what it has seen — so they are serialized with the agent and
must be given to ``Core`` as core preprocessors, or to the agent as agent preprocessors, depending on whether the
stored dataset should hold the raw or the transformed observation.

The two differ in when their statistics advance. A core preprocessor is updated by ``Core`` on every step, during
evaluation as well as training, and the dataset stores what it returns. An agent preprocessor leaves the dataset
raw and is held by the :class:`~mushroom_rl.core.history_manager.HistoryManager`, which applies it to every
observation used as policy input and updates its statistics once per ``fit``; see
:doc:`../../tutorials/tutorials.8_history_manager`.

They also differ in the array backend they see. A core preprocessor is applied to the environment's
observations and an agent preprocessor to the agent's, which need not be the same backend. The ``backend``
argument of the constructor selects it, defaulting to the MDP one, so an agent preprocessor built for an
agent whose backend differs from the environment's must be given it explicitly::

    agent.add_agent_preprocessor(StandardizationPreprocessor(mdp.info, backend='torch'))

.. automodule:: mushroom_rl.rl_utils.preprocessors
    :private-members:
