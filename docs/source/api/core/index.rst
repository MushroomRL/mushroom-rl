Core
====

The three basic interface of mushroom_rl are the Agent, the Environment and the Core interface.

- The ``Agent`` is the basic interface for any Reinforcement Learning algorithm.
- The ``Environment`` is the basic interface for every problem/task that the agent should solve.
- The ``Core`` is a class used to control the interaction between an agent and an environment.

We provide the logging functionality with the ``Logger`` class. Finally, the ``MushroomObject`` interface implements
serialization of MushroomRL data on the disk (load/save functionality), forwards a logger down the object tree, and
names its objects through the ``name`` and ``full_name`` methods.

The data collected while the agent interacts with the environment is stored in a ``Dataset``, and the array type used
to store it is selected per component through the ``backend`` declared in ``MDPInfo`` and in the agent constructor;
``ArrayBackend`` implements the conversions between them.

.. autosummary::
   :nosignatures:

   ~mushroom_rl.core.agent.Agent
   ~mushroom_rl.core.environment.Environment
   ~mushroom_rl.core.environment.MDPInfo
   ~mushroom_rl.core.vectorized_env.VectorizedEnvironment
   ~mushroom_rl.core.multiprocess_environment.MultiprocessEnvironment
   ~mushroom_rl.core.core.Core
   ~mushroom_rl.core.dataset.Dataset
   ~mushroom_rl.core.dataset.VectorizedDataset
   ~mushroom_rl.core.extra_info.ExtraInfo
   ~mushroom_rl.core.history_manager.HistoryManager
   ~mushroom_rl.core.array_backend.ArrayBackend
   ~mushroom_rl.core.mushroom_object.MushroomObject
   ~mushroom_rl.core.spaces.Box
   ~mushroom_rl.core.spaces.Discrete
   ~mushroom_rl.core.logger.Logger

.. toctree::
   :maxdepth: 1

   agent
   environment
   vectorized_environment
   core
   dataset
   extra_info
   history_manager
   array_backend
   mushroom_object
   spaces
   logger/index
