MuJoCo
======

The environments simulated with `MuJoCo <https://mujoco.org/>`_. ``MuJoCo`` is the base class: it loads an XML
model, builds the observation from a specification of the model entities to read, and leaves the reward and the
termination condition to the subclass. ``MultiMuJoCo`` loads several models at once and samples one of them at the
beginning of every episode, which is how domain randomization over the model is expressed.

Writing a new MuJoCo environment means subclassing ``MuJoCo`` and describing the observation with the helpers in
:doc:`../../utils/mujoco`.

.. autosummary::
   :nosignatures:

   ~mushroom_rl.environments.mujoco.MuJoCo
   ~mushroom_rl.environments.mujoco.MultiMuJoCo

.. toctree::
   :maxdepth: 1

   mujoco
   locomotion
   manipulation
   air_hockey
   ball_in_a_cup
