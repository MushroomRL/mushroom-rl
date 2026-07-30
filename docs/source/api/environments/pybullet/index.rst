PyBullet
========

The environments simulated with `PyBullet <https://pybullet.org>`_. ``PyBullet`` is the base class: it loads the
models, drives the joints through the helpers in :doc:`../../utils/pybullet`, and builds the observation from a
specification of the bodies and joints to read.

.. autosummary::
   :nosignatures:

   ~mushroom_rl.environments.pybullet.PyBullet

.. toctree::
   :maxdepth: 1

   pybullet
   air_hockey
