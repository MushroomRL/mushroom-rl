MushroomObject
==============

``MushroomObject`` is the common base of the MushroomRL objects, providing two cross-cutting capabilities.
- Serialization: a subclass declares which attributes to persist with ``_add_save_attr``, and ``save``/``load`` walk
that declaration to write the object to, and rebuild it from, a zip file.
- Logger forwarding: a subclass declares its loggable children with ``_add_logger_attr``, and ``set_logger`` forwards a
logger down the whole object tree, composing the metric names into a group hierarchy.

.. automodule:: mushroom_rl.core.mushroom_object
    :private-members:
