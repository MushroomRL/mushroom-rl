Array backend
=============

``ArrayBackend`` is a namespace facade over the array libraries MushroomRL supports — numpy, PyTorch, and a plain
list backend — exposing one common set of operations so that the rest of the library is written once and runs on
any of them. The backend of a component is selected by name (``'numpy'``, ``'torch'``, ``'list'``), for example
through the ``backend`` argument of ``MDPInfo`` for an environment, or of an agent. Conversions between backends
happen at the agent boundary.

.. automodule:: mushroom_rl.core.array_backend
    :private-members:
