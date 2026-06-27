The MushroomObject Interface
============================

In this tutorial, we explain in detail the ``MushroomObject`` interface, i.e. the common base class of almost every
MushroomRL component: agents, policies, approximators, parameters, distributions, environments, and datasets all derive
from it. It exists to give these heterogeneous objects two cross-cutting capabilities for free, so that their own code can
stay focused on the reinforcement learning logic. The first one is serialization: an object can be saved to and loaded
from disk in a robust, native format, declaring exactly which of its attributes to persist and how. The second one is
logger forwarding: a logger attached to a top-level object (typically an agent) is automatically propagated down its tree
of sub-objects, so that each component can log its own quantities under a coherent hierarchy of metric names, without
holding a reference to the logger in advance.

Both capabilities rely on the same idea: a ``MushroomObject`` declares its relevant attributes and loggable children
(with ``_add_save_attr`` and ``_add_logger_attr``), and the base class walks that declaration to recursively save the
object or forward the logger. Because the wiring is data rather than hand-written code, it composes naturally through
nested objects, survives inheritance and reload, and a subclass only needs to declare what it adds.

We first explain how to use classes implementing the ``MushroomObject`` interface, and then we provide a small example of
how to implement it on a custom class to serialize the object properly on disk. Finally, we describe how the same
interface is used to forward a logger down the object tree.

The Mushroom RL save format (extension ``.msh``) is nothing else than a zip file, containing some information (stored into
the ``config`` file) to load the object. This information can be accessed easily and you can try to recover the information
by hand from corrupted files.

Note that it is always possible to serialize Python objects with the pickle library. However, the MushroomRL
serialization interface use a native format, is easy to use, and is more robust to code changes, as it doesn't serialize
the entire class, but only the data. Furthermore, it is possible to avoid the serialization of some class variables,
such as shared objects or big arrays, e.g. replay memories.

Save and load from disk
-----------------------

All the algorithms, policies, approximators, and parameters implemented in MushroomRL derive from
``MushroomObject``, and therefore can be saved and loaded out of the box.

As an example, we save a MushroomRL ``Parameter`` on disk. We create the parameter and then we serialize it to disk
using the ``save`` method of the ``MushroomObject`` class:

.. code-block:: python

    from mushroom_rl.rl_utils.parameters import Parameter

    parameter = Parameter(1.0)
    print('Initial parameter value: ', parameter())
    parameter.save('parameter.msh')

This code creates a ``parameters.msh`` file in the working directory.

You can also specify a directory:

.. code-block:: python

    from pathlib import Path
    base_dir = Path('tmp')
    file_name = base_dir / 'parameter.msh'
    parameter.save(file_name)

This create a ``tmp`` folder (if it doesn't exist) in the working directory and save the ``parameters.msh`` file
inside it.

Now, we can set another value for our parameter variable:

.. code-block:: python

    parameter = Parameter(0.5)
    print('Modified parameter value: ', parameter())

Finally, we load the previously stored parameter to go back to the previous state using the ``load`` method:

.. code-block:: python

    parameter = Parameter.load('parameter.msh')
    print('Loaded parameter value: ', parameter())

You can also call the load directly from the MushroomObject class:

.. code-block:: python

    from mushroom_rl.core import MushroomObject
    parameter = MushroomObject.load('parameter.msh')
    print('Loaded parameter value (MushroomObject): ', parameter())

The same approach can be used to save an agent, a policy, or an approximator.

Full Save
---------

The ``save`` method has an optional ``full_save`` flag, which by default is set to False. In the previous parameter
example, this flag does not affect. However, when saving a Reinforcement Learning algorithm or other complex
objects, setting this flag to true forces the agent to save data structures that are normally excluded from a save
file, such as the replay memory in DQN.

This implementation choice avoids large save files for agents with huge data structures, and allows to avoid storing
duplicated information (such as the Q function of in epsilon greedy policy, when saving the algorithm).
The ``full_save`` instead, enforces a complete serialization of the agent, retaining all the information.

Implementing the MushroomObject interface
-----------------------------------------

We give a simple example of how to implement the ``MushroomObject`` interface in MushroomRL for a custom class.
We use almost all possible data persistence types implemented.

We start the example by importing the ``MushroomObject`` interface, the torch library, the NumPy library, and the MushroomRL
``Parameter class``.

.. literalinclude:: code/serialization.py
   :lines: 1-5

While it is required to import the ``MushroomObject`` interface, the other three imports are only required by this example, as
they are used to create variables of such type.

Now we define a class implementing the ``MushroomObject`` interface. In this case, we call it ``TestClass``.
In the constructor we first build a set of variables of different types, and then we specify which variables
we want to be saved in the MushroomRL file passing keywords to the ``self._add_save_attr`` method. Note that
``MushroomObject`` has no constructor to call: the important attributes are set up automatically when the
object is created.

.. literalinclude:: code/serialization.py
   :lines: 8-42

Some remarks about the ``self._add_save_attr`` method: the keyword name must be the name of the variable we want to
store in the file, while the associated value is the method we want to use to store such variables.

The available methods are:

- **primitive**, to store any primitive type. This includes lists and dictionaries of primitive values.
- **mushroom**, to store any type implementing the ``MushroomObject`` interface. Also, lists of ``MushroomObject`` instances are supported.
- **numpy**, to store NumPy arrays.
- **torch**, to store any torch object.
- **pickle**, to store any Python object that cannot be stored with the above methods.
- **json**, can be used if you need a textual output version, that is easy to read.

Another important aspect to remember is that any method can be ended by a '**!**', to specify that the field must be
serialized if and only if the ``full_save`` flag is set to true.

To conclude the implementation of our ``MushroomObject`` interface, we might want to implement also the
``self._post_load`` method. This method is executed after all the data specified in ``self._add_save_attr`` has been
loaded into the class. It can be useful to set the variables not saved in the file to a default variable.

.. literalinclude:: code/serialization.py
   :lines: 44-48

In this scenario, we have to set the ``self.not_important`` variable to his default value, but only if it's None, i.e.
has not been loaded from the file, because the file didn't contain it.
Also, we set the ``self._list_reference`` variable to maintain its original semantic, i.e. to contain a
reference to the content of the ``self._dictionary`` variable.

To test the implementation, we write a function to write in easy to read way the content of the class:

.. literalinclude:: code/serialization.py
   :lines: 51-60

Finally, we test the save functionality with the following code:

.. literalinclude:: code/serialization.py
   :lines: 63-

We can see that the content of ``self.not_important`` is stored only if the ``full_save`` flag is set to true.

The last remark is that the ``MushroomObject`` interface works also in presence of inheritance. If you extend a
``MushroomObject`` subclass, you only need to add the new attributes defined by the child class.

Forwarding a logger through the MushroomObject interface
--------------------------------------------------------

Besides save and load, the ``MushroomObject`` interface also takes care of distributing a logger across the
object tree used by the algorithms. A logger is attached to an object with ``set_logger`` and is
automatically forwarded to the loggable children declared by the object, so that the relevant quantities of
an object and of its sub-objects are logged under a hierarchy of grouped metric names.

The loggable children are declared with ``self._add_logger_attr``, in the same spirit as
``self._add_save_attr``: the first argument is an optional group prefix shared by the registered children,
attributes passed positionally use their default metric name, and attributes passed as keywords use an
explicit metric name. For example, an algorithm registers its critic approximator and its exploration
parameter as:

.. code-block:: python

    self._add_logger_attr('_V', group='critic')                 # logs critic/loss
    self._add_logger_attr(_epsilon='epsilon', group='policy')   # logs policy/epsilon

When ``set_logger`` is called (typically by the ``Core``, see the Logger tutorial), the logger is stored
and forwarded to each registered child, joining the child group to the current prefix, so the hierarchy is
composed automatically down the object tree. As for ``_add_save_attr``, children are referenced by name, so
the forwarding keeps working after an object is reassigned or loaded from disk (the registry is part of the
saved data). A single value is logged with ``self._logger.log_training(...)`` from inside the object, where
the first positional argument is the metric group and the values are passed as keyword arguments.
