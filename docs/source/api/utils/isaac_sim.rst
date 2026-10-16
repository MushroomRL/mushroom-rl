Isaac Sim utils
===============

The helpers used to write an Isaac Sim environment. ``IsaacLauncher`` starts the simulator, and has to be called
before anything else in the Isaac Sim layer can be imported. The others return batched quantities, one entry per
parallel environment: ``SceneBuilder`` assembles the stage and clones it once per environment, ``ObservationHelper``
reads and writes the properties of the simulated entities and assembles the batched observation out of them,
``ActuationHelper`` drives the joints the agent controls in the mode an ``ActuationType`` names, ``CollisionHelper``
reports the contacts of the tracked bodies, ``IsaacGpuParams`` sets the GPU buffer sizes of the physics scene, and
``IsaacViewer`` holds the camera looking at the scene.

.. automodule:: mushroom_rl.utils.isaac_sim.launcher
    :private-members:

.. automodule:: mushroom_rl.utils.isaac_sim.scene_builder
    :private-members:

.. automodule:: mushroom_rl.utils.isaac_sim.gpu_params
    :private-members:

.. automodule:: mushroom_rl.utils.isaac_sim.observation_helper
    :private-members:

.. automodule:: mushroom_rl.utils.isaac_sim.actuation_helper
    :private-members:

.. automodule:: mushroom_rl.utils.isaac_sim.collision_helper
    :private-members:

.. automodule:: mushroom_rl.utils.isaac_sim.viewer
    :private-members:
