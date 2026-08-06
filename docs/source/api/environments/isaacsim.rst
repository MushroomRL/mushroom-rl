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
:doc:`Isaac Sim utils <../utils/isaac_sim>`, which must be called before the modules below can be imported. As
simulation backend, we rely on the Isaac Sim 6.0 default, PhysX; the launcher can select Newton instead.

``IsaacSim`` is the base class of every environment below. It owns the cloned scenes and builds the batched
observation from the specification of the articulation entities to read. ``QuadrupedIsaac`` is a further base
class, shared by the three legged robots below, holding the command-tracking logic and observations common to
all of them; ``SilverBadgerIsaac`` in turn subclasses ``HoneyBadgerIsaac``, differing only in its robot-specific
configuration.

.. autosummary::
   :nosignatures:

   ~mushroom_rl.environments.isaacsim_env.IsaacSim
   ~mushroom_rl.environments.isaacsim_envs.cartpole.CartPoleIsaac
   ~mushroom_rl.environments.isaacsim_envs.quadruped.QuadrupedIsaac
   ~mushroom_rl.environments.isaacsim_envs.a1.A1Isaac
   ~mushroom_rl.environments.isaacsim_envs.honey_badger.HoneyBadgerIsaac
   ~mushroom_rl.environments.isaacsim_envs.silver_badger.SilverBadgerIsaac

.. automodule:: mushroom_rl.environments.isaacsim_env
    :private-members:

.. automodule:: mushroom_rl.environments.isaacsim_envs.cartpole

.. automodule:: mushroom_rl.environments.isaacsim_envs.quadruped
    :private-members:

.. automodule:: mushroom_rl.environments.isaacsim_envs.a1

.. automodule:: mushroom_rl.environments.isaacsim_envs.honey_badger

.. automodule:: mushroom_rl.environments.isaacsim_envs.silver_badger
