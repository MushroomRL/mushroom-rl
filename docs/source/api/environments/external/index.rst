External libraries
==================

Wrappers exposing the environments of other libraries through the MushroomRL ``Environment`` interface. Each of
them is registered only if its dependency imports successfully, so installing the corresponding extra is what makes
the environment appear in the registry.

.. autosummary::
   :nosignatures:

   ~mushroom_rl.environments.atari.Atari
   ~mushroom_rl.environments.gymnasium_env.Gymnasium
   ~mushroom_rl.environments.dm_control_env.DMControl
   ~mushroom_rl.environments.minigrid_env.MiniGrid
   ~mushroom_rl.environments.minigrid_env.MiniGridRGB

.. toctree::
   :maxdepth: 1

   atari
   gymnasium
   dm_control
   minigrid
