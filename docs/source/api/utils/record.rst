Record
======

The video recorders the ``VideoLogger`` writes through. ``VideoRecorder`` dispatches on whether the frames come
from a single environment or from a vectorized one, in which case the copies are concatenated into a single video.

.. automodule:: mushroom_rl.utils.record
    :private-members:
