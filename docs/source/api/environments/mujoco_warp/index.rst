MuJoCo Warp
===========

The environments simulated with `MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp>`_, the GPU
implementation of MuJoCo. These environments are ``VectorizedEnvironment`` subclasses: they step a batch of
simulated copies at once, and are driven through the vectorized ``Core``. They are registered with
:meth:`~mushroom_rl.core.Environment.make`, and are installed with the ``mujoco_warp`` extra.

``MuJoCoWarp`` is the base class of every environment. It loads an XML model like :doc:`MuJoCo <../mujoco/index>`
does, and its observation is described with the ``WarpObservationHelper`` documented with the other
:doc:`MuJoCo utils <../../utils/mujoco>`. These are the environments that can be instantiated:

.. autosummary::
   :nosignatures:

   ~mushroom_rl.environments.mujoco_warp_envs.locomotion.ant.AntWarp
   ~mushroom_rl.environments.mujoco_warp_envs.locomotion.half_cheetah.HalfCheetahWarp
   ~mushroom_rl.environments.mujoco_warp_envs.locomotion.hopper.HopperWarp
   ~mushroom_rl.environments.mujoco_warp_envs.locomotion.walker_2d.Walker2DWarp

The environments and their base class are documented on the following pages:

.. toctree::
   :maxdepth: 1

   mujoco_warp
   locomotion
