How to use the Mushroom Serialization
=====================================

In this tutorial, we will explain in detail the ``Serializable`` interface. We will first explain how to use classes
implementing the ``Serializable`` interface, and then we will provide a small example of how to implement the
``Serializable`` interface on a custom class to serialize the object properly on disk.

The Mushroom RL save format (extension ``.msh``) is nothing else than a zip file, containing some information (stored into
the ``config`` file) to load the system. This information can be accessed easily and you can try to recover the information
by hand from corrupted files.

Note that it is always possible to serialize Python objects with the pickle library. However, the MushroomRL
serialization interface use a native format, is easy to use, and is more robust to code changes, as it doesn't serialize
the entire class, but only the data. Furthermore, it is possible to avoid the serialization of some class variables,
such as shared objects or big arrays, e.g. replay memories.

Save and load from disk
-----------------------

Many mushroom objects implement the serialization interface. All the algorithms, policies, approximators, and parameters
implemented in MushroomRL use the ``Serializable`` interface.

As an example, we will save a MushroomRL ``Parameter`` on disk. We create the parameter and then we serialize it to disk
using the ``save`` method of the serializable class:
.. code-block:: python

    from mushroom_rl.utils.parameters import Parameter

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

This will create a ``tmp`` folder (if it doesn't exist) in the working directory and save the ``parameters.msh`` file
inside it.

Now, we can set another value for our parameter variable:
.. code-block:: python

    parameter = Parameter(0.5)
    print('Modified parameter value: ', parameter())

Finally, we load the previously stored parameter to go back to the previous state using the ``load`` method:
.. code-block:: python

    parameter = Parameter.load('parameter.msh')
    print('Loaded parameter value: ', parameter())

You can also call the load directly from the Serializable class:
.. code-block:: python

    from mushroom_rl.core import Serializable
    parameter = Serializable.load('parameter.msh')
    print('Loaded parameter value (Serializable): ', parameter())

The same approach can be used to save an agent, a policy, or an approximator.

Full Save
---------

The ``save`` method has an optional ``full_save`` flag, which by default is set to False. In the previous parameter
example, this flag will not affect. However, when saving a Reinforcement Learning algorithm or other complex
objects, setting this flag to true will force the agent to save data structures that are normally excluded from a save
file, such as the replay memory in DQN.

This implementation choice avoids large save files for agents with huge data structures, and allows to avoid storing
duplicated information (such as the Q function of in epsilon greedy policy, when saving the algorithm).
The ``full_save`` instead, enforces a complete serialization of the agent, retaining all the information.

Implementing the Serializable interface
---------------------------------------

We will give a simple example of how to implement the ``Serializable`` interface in MushroomRL for a custom class.

We will try to use almost all possible data persistence types implemented.
We start the example by importing the serializable interface, the torch library, the NumPy library, and the MushroomRL
``Parameter class``.

.. literalinclude:: code/serialization.py
   :lines: 1-5

While it is required to import the ``Serializable`` interface, the other three imports are only required by this example, as
they will be used to create variables of such type.

Now we define a class implementing the ``Serializable`` interface. In this case, we call it ``TestClass``.
The constructor can be divided into two parts: first, we build a set of variables of different types.
Then, we call the superclass constructor, i.e. the constructor of ``Serializable``. Finally, we specify which variables
we want to be saved in the mushroom file passing keywords to the ``self._add_save_attr`` method.

.. literalinclude:: code/serialization.py
   :lines: 8-45

Some remarks about the ``self._add_save_attr`` method: the keyword name must be the name of the variable we want to
store in the file, while the associated value is the method we want to use to store such variables.

The available methods are:

- **primitive**, to store any primitive type. This includes lists and dictionaries of primitive values.
- **mushroom**, to store any type implementing the Serializable interface. Also, lists of serializable objects are supported.
- **numpy**, to store NumPy arrays.
- **torch**, to store any torch object.
- **pickle**, to store any Python object that cannot be stored with the above methods.
- **json**, can be used if you need a textual output version, that is easy to read.

Another important aspect to remember is that any method can be ended by a '**!**', to specify that the field must be
serialized if and only if the ``full_save`` flag is set to true.

To conclude the implementation of our ``Serializable`` interface, we might want to implement also the
``self._post_load`` method. This method is executed after all the data specified in ``self._add_save_attr`` has been
loaded into the class. It can be useful to set the variables not saved in the file to a default variable.

.. literalinclude:: code/serialization.py
   :lines: 47-51

In this scenario, we have to set the ``self.not_important`` variable to his default value, but only if it's None, i.e.
has not been loaded from the file, because the file didn't contain it.
Also, we set the `` self._list_primitive`` variable to maintain its original semantic, i.e. to contain a
reference to the content of the ``self._dictionary`` variable.

To test the implementation, we write a function to write in easy to read way the content of the class:

.. literalinclude:: code/serialization.py
   :lines: 54-63

Finally, we test the save functionality with the following code:

.. literalinclude:: code/serialization.py
   :lines: 66-

We can see that the content of ``self.not_important`` is stored only if the ``full_save`` flag is set to true.

The last remark is that the ``Serializable`` interface works also in presence of inheritance. If you extend a
serializable class, you only need to add the new attributes defined by the child class.