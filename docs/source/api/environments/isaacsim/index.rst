Isaac Sim
=========

The environments simulated with `Isaac Sim <https://developer.nvidia.com/isaac-sim>`_.
These environments are ``VectorizedEnvironment`` subclasses. Differently from all other environments, they are
**not** registered with :meth:`~mushroom_rl.core.Environment.make` and must be imported from their module. Isaac Sim
is not part of the ``all`` extra, and is installed with the ``isaacsim`` one.

This interface targets **Isaac Sim 6.1**. The simulator is started by
:class:`~mushroom_rl.utils.isaac_sim.launcher.IsaacLauncher`, documented with the other
:doc:`Isaac Sim utils <../../utils/isaac_sim>`, which must be called before any Isaac Sim environment can be imported.
The default physics engine is PhysX; the launcher can select Newton instead.

``IsaacSim`` is the base class of every environment, and
:class:`~mushroom_rl.environments.isaacsim_envs.quadruped.QuadrupedIsaac` a further one shared by the legged
robots. These are the environments that can be instantiated:

.. autosummary::
   :nosignatures:

   ~mushroom_rl.environments.isaacsim_envs.cartpole.CartPoleIsaac
   ~mushroom_rl.environments.isaacsim_envs.a1.A1Isaac
   ~mushroom_rl.environments.isaacsim_envs.go2.Go2Isaac
   ~mushroom_rl.environments.isaacsim_envs.honey_badger.HoneyBadgerIsaac
   ~mushroom_rl.environments.isaacsim_envs.silver_badger.SilverBadgerIsaac

The environments, their base classes and the components of the quadruped tasks are documented on the following
pages:

.. toctree::
   :maxdepth: 1

   isaacsim
   quadruped
   quadruped_randomization
   quadruped_commands
   quadruped_robots
