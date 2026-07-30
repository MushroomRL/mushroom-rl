Utils
=====

General-purpose helpers, not specific to reinforcement learning. Unlike :doc:`../rl_utils/index`, nothing here
knows about agents, environments or datasets: these are the mathematical, plotting, recording and simulator-side
utilities the rest of the library is written on top of.

The last three pages are the helpers used to *write* an environment on a given simulator, and are needed when
subclassing ``MuJoCo``, ``PyBullet`` or ``IsaacSim`` rather than when using one of the environments already
implemented.

.. toctree::
   :maxdepth: 1

   math
   experiments
   features
   record
   torch_utils
   viewer
   callbacks
   plots
   mujoco
   pybullet
   isaac_sim
