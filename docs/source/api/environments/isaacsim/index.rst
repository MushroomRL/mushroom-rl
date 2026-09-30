Isaac Sim
=========

The environments simulated with `Isaac Sim <https://developer.nvidia.com/isaac-sim>`_.
These environments are ``VectorizedEnvironment`` subclasses, and differently from all other environments, they are
**not** registered: ``mushroom_rl/environments/__init__.py`` does not import ``isaacsim_envs``, so
:meth:`~mushroom_rl.core.Environment.make` does not know them and they must be imported from their module
directly. Isaac Sim is not one of the extras declared in ``setup.py`` either, and has to be installed separately.

This interface targets **Isaac Sim 6.0** and its supported ``isaacsim.core.experimental.*`` API, which returns Warp
arrays and is only available once the simulation app is running. The app is started by
:class:`~mushroom_rl.utils.isaac_sim.launcher.IsaacLauncher`, documented with the other
:doc:`Isaac Sim utils <../../utils/isaac_sim>`, which must be called before the modules below can be imported. As
simulation backend, we rely on the Isaac Sim 6.0 default, PhysX; the launcher can select Newton instead.

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

The legged robots, and the domain randomization they are trained under, are documented on their own page:

.. toctree::
   :maxdepth: 1

   quadruped

.. rubric:: Interface

The base class of every Isaac Sim environment. It owns the cloned scenes, steps the simulation for the number of
intermediate steps declared, and assembles the batched observation from the specification of the articulation
entities it is given, leaving the reward and the termination condition to the subclass.

.. automodule:: mushroom_rl.environments.isaacsim_env
    :private-members:

.. rubric:: Cart pole

The batched cart pole, the smallest Isaac Sim environment: a single actuated slider carrying a free pole, used to
check the interface end to end without the cost of a full robot.

.. automodule:: mushroom_rl.environments.isaacsim_envs.cartpole
