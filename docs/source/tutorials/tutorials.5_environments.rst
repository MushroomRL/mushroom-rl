How to use the Environment interface
====================================

Here we explain in a bit of detail the usage of the MushroomRL Environment interface.
First, we explain how to use the registration interface. The registration enables the construction of environments
from string specification. Then we construct a toy environment to show how it is possible to add new MushroomRL
environments.


Old-school enviroment creation
------------------------------

In MushroomRL, environments are simply class objects that extend the environment interface.
To create an environment, you can simply call its constructor.
You can build the Segway environment as follows:

.. code-block:: python

    from mushroom_rl.environments import Segway

    env = Segway()

Some environments may have a constructor which is too low level, and you may want to generate a vanilla version of it
using as few parameters as possible.
An example is the Linear Quadratic Regulator (LQR) environment, which requires a set of matrices to define the linear
dynamics and the quadratic cost function. To provide an easier interface, the ``generate`` class method is exposed.To
generate a simple 3-dimensional LQR problem, with Identity transition and action matrices, and a trivial
quadratic cost function, you can use:


.. code-block:: python

    from mushroom_rl.environments import LQR

    env = LQR.generate(dimensions=3)

See the documentation of ``LQR.generate`` to know all the available parameters and effects.

Environment registration
------------------------

From version 1.7.0, it is possible to register MushroomRL environments and build the environment by specifying only
the name.

You can list the registered environments as follows:

.. code-block:: python

    from mushroom_rl.core import Environment

    env_list = Environment.list_registered()
    print(env_list)

Every registered environment can be build using the name.
For example, to create the SheepSteering environment you can use:

.. code-block:: python

    env = Environment.make('ShipSteering')

To build environments, you may need to pass additional parameters.
An example of this is the ``Gym`` environment which wraps most OpenAI Gym environments, except the Atari ones, which
uses the ``Atari`` environment to implement proper preprocessing.

If you want to build the ``Pendulum-v0`` gym environment you need to pass the environment name as a parameter:

.. code-block:: python

    env = Environment.make('Gym', 'Pendulum-v0')

However, for environments that are interfaces to other libraries such as ``Gym``, ``Atari`` or ``DMControl`` a notation
with a dot separator is supported. For example to create the pendulum you can also use:

.. code-block:: python

    env = Environment.make('Gym.Pendulum-v0')

Or, to create the ``hopper`` environment with ``hop`` task from DeepMind control suite you can use:

.. code-block:: python

    env = Environment.make('DMControl.hopper.hop')


If an environment implements the generate method, it will be used to build the environment instead of the constructor.
As the generate method is higher-level interface w.r.t. the constructor, it will require less parameters.

To generate the 3-dimensional LQR problem mentioned in the previous section you can use:

.. code-block:: python

    env = Environment.generate('LQR', dimensions=3)


Finally, you can register new environments. Suppose that you have created the environment class ``MyNewEnv``, which
extends the base ``Environment`` class. You can register the environment as follows:

.. code-block:: python

    MyNewEnv.register()

You can put this line of code after the class declaration, or in the ``__init__.py`` file of your library.
If you do so, the first time you import the file, you will register the environment. Notice that this registration is
not saved on disk, thus, you need to register the environment every time the python interpreter is executed.

Creating a new environment
--------------------------

We show you an example of how to construct a MushroomRL environment.
We create a simple room environment, with discrete actions, continuous state space, and mildly stochastic dynamics.
The objective is to move the agent from any point of the room towards the goal point. The agent takes a penalty at every
step equal to the distance to the objective. When the agent reaches the goal the episode ends. The agent can move in the
room by using one of the 4 discrete actions, North, South, West, East.

First of all, we import all the required classes: NumPy for working with the array, the Environment interface and
the MDPInfo structure, which contains the basic information about the Environment.

Given that we are implementing a simple visualization function, we import also the viewer class, which is a Pygame
wrapper, that can be used to render easily RL environments.


.. literalinclude:: code/room_env.py
   :lines: -6

Now, we can create the environment class.

We first extend the environment class and create the constructor:

.. literalinclude:: code/room_env.py
   :lines: 9-34

It's important to notice that the superclass constructor needs the information stored in the ``MDPInfo`` structure.
This structure contains the action and observation space, the discount factor ``gamma``, and the horizon.
The horizon is used to cut the trajectories when they are too long. When the horizon is reached the episode is
terminated, however, the state might not be absorbing. The absorbing state flag is explicitly set in the environment step
function.
Also, notice that the ``Environment`` superclass has no notion of the environment state, so we need to store it by
ourselves. That's why we create the ``self._state`` variable and we initialize it to ``None``.
Other environment information such as the goal position and area is stored into class variables.

Now we implement the reset function. This function is called at the beginning of every episode. It's possible to force
the initial state. For this reason, we have to manage two scenarios: when the initial state is given and when it is set
to None. If the initial state is not given, we sample randomly among the valid states.


.. literalinclude:: code/room_env.py
   :lines: 36-52

Now it's time to implement the step function, that specifies the transition function of the environment, computes the
reward, and signal absorbing states, i.e. states where every action keeps you in the same state, achieving 0 reward.
When reaching the absorbing state we cut the trajectory, as their value function is always 0, and no further exploration
is needed.

.. literalinclude:: code/room_env.py
   :lines: 54-87

Finally, we implement the render function using our ``Viewer`` class. This class wraps Pygame to provide an easy
visualization tool for 2D Reinforcement Learning algorithms. The viewer class has many functionalities, but here we
simply draw two circles representing the agent and the goal area:

.. literalinclude:: code/room_env.py
   :lines: 89-97

For more information about the viewer, refer to the class documentation.

To conclude our environment, it's also possible to register it as specified in the previous section of this tutorial:

.. literalinclude:: code/room_env.py
   :lines: 100-101


Learning in the toy environment
-------------------------------

Now that we have created our environment, we try to solve it using Reinforcement Learning. The following code uses the
True Online SARSA-Lambda algorithm, exploiting a tiles approximator.

We first import all necessary classes and utilities, then we construct the environment (we set the seed for
reproducibility).

.. literalinclude:: code/room_env.py
   :lines: 103-116

We now proceed then to create the agent policy, which is a linear policy using tiles features, similar
to the one used by the Mountain Car experiment from R. Sutton book.

.. literalinclude:: code/room_env.py
   :lines: 118-139

Finally, using the ``Core`` class we set up an RL experiment. We first evaluate the initial policy for three episodes on the
environment. Then we learn the task using the algorithm build above for 20000 steps.
In the end, we evaluate the learned policy for 3 more episodes.

.. literalinclude:: code/room_env.py
   :lines: 141-
