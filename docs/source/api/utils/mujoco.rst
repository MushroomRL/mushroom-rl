MuJoCo utils
============

The helpers used to write a MuJoCo environment. ``ObservationHelper`` turns a declarative specification — a list of
``(name, entity, ObservationType)`` entries — into the code that reads those entities out of the simulator data
and concatenates them into the observation vector, so a subclass of ``MuJoCo`` declares its observation instead of
indexing the raw arrays. ``MujocoViewer`` renders the scene, and ``forward_kinematics`` evaluates the pose of a
body for a given joint configuration.

.. automodule:: mushroom_rl.utils.mujoco.observation_helper
    :private-members:

.. automodule:: mushroom_rl.utils.mujoco.kinematics

.. automodule:: mushroom_rl.utils.mujoco.viewer
    :private-members:
