Extra info
==========

``ExtraInfo`` collects the additional per-step information an environment or an agent returns beside the transition
itself, keeping one column per key and growing it as samples arrive, so that it can be stored inside a ``Dataset``
and read back in the same array backend as the rest of the data.

.. automodule:: mushroom_rl.core.extra_info
