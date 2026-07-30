Isaac Sim
=========

The environments simulated with `Isaac Sim <https://developer.nvidia.com/isaac-sim>`_. Unlike every other
environment in MushroomRL, they are ``VectorizedEnvironment`` subclasses: the simulator steps a batch of copies at
once, so the observations, actions and rewards carry a leading environment axis and the agent is driven through
the vectorized ``Core``.

They are **not** registered: ``mushroom_rl/environments/__init__.py`` does not import ``isaacsim_envs``, so
:meth:`~mushroom_rl.core.Environment.make` does not know them and they must be imported from their module
directly. Isaac Sim is not one of the extras declared in ``setup.py`` either, and has to be installed separately.

.. autosummary::
   :nosignatures:

   ~mushroom_rl.environments.isaacsim_env.IsaacSim
   ~mushroom_rl.environments.isaacsim_envs.cartpole.CartPole
   ~mushroom_rl.environments.isaacsim_envs.a1_walking.A1Walking
   ~mushroom_rl.environments.isaacsim_envs.honey_badger_walking.HoneyBadgerWalking
   ~mushroom_rl.environments.isaacsim_envs.silver_badger_walking.SilverBadgerWalking

.. toctree::
   :maxdepth: 1

   isaacsim_env
   cartpole
   a1_walking
   honey_badger_walking
   silver_badger_walking
