How to make a deep RL experiment
================================

The usual script to run a deep RL experiment does not significantly differ from
the one for a shallow RL experiment.
This tutorial shows how to solve `Atari <https://gym.openai.com/envs/#atari/>`_
games in MushroomRL using ``DQN``, and how to solve
`MuJoCo <https://github.com/deepmind/dm_control/>`_ tasks using ``DDPG``. This
tutorial will not explain some technicalities that are already described in the
previous tutorials, and will only briefly explain how to run deep RL experiments.
Be sure to read the previous tutorials before starting this one.

Solving Atari with DQN
----------------------
This script runs the experiment to solve the Atari Breakout game as described in
the DQN paper *"Human-level control through deep reinforcement learning", Mnih V. et
al., 2015*). We import ``AtariNetwork`` from the networks package and define
a few helper functions:

.. literalinclude:: code/dqn.py
   :lines: 1-24

We then set some hyperparameters and create the ``mdp`` and the policy ``pi``.
Differently from the literature, we use ``Adam`` as the optimizer:

.. literalinclude:: code/dqn.py
   :lines: 27-52

Then, the ``approximator``:

.. literalinclude:: code/dqn.py
   :lines: 54-66

Finally, the ``agent`` and the ``core``:

.. literalinclude:: code/dqn.py
   :lines: 68-83

Eventually, the learning loop is performed. As done in literature, learning and
evaluation steps are alternated:

.. literalinclude:: code/dqn.py
   :lines: 87-107

Solving MuJoCo with DDPG
------------------------
This script runs the experiment to solve the Walker-Stand MuJoCo task, as
implemented in `MuJoCo <https://github.com/deepmind/dm_control/>`_. We import
``ActorNetwork`` and ``CriticNetwork`` from the networks package.

We create the ``mdp``, the policy, and set some hyperparameters:

.. literalinclude:: code/ddpg.py
   :lines: 1-28

Note that the policy is not instantiated in the script, since in DDPG the
instantiation is done inside the algorithm constructor.

We create the actor and the critic approximators. The critic takes ``state`` and ``action`` as two
separate inputs, so its ``input_shape`` is a list of the two shapes:

.. literalinclude:: code/ddpg.py
   :lines: 30-47

Finally, we create the ``agent`` and the ``core``:

.. literalinclude:: code/ddpg.py
   :lines: 49-59

As in ``DQN``, we alternate learning and evaluation steps:

.. literalinclude:: code/ddpg.py
   :lines: 61-76
