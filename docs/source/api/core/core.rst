Core
====

``Core`` runs the interaction loop between an agent and an environment: it calls ``draw_action`` on the agent and
``step`` on the environment, applies the core preprocessors, collects the samples into a ``Dataset``, and calls
``fit`` on the agent according to the step or episode counts given to ``learn``. The same class drives both the
single-environment and the vectorized loop, dispatching on the environment it is built with.

.. automodule:: mushroom_rl.core.core
