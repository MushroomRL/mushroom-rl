How to make a deep RL experiment
================================

The usual script to run a deep RL experiment does not significantly differ from
the one for a shallow RL experiment.
This tutorial shows how to solve `Atari <https://gym.openai.com/envs/#atari/>`_
games in Mushroom using ``DQN``, and how to solve
`MuJoCo <https://github.com/deepmind/dm_control/>`_ tasks using ``DDPG``. This
tutorial will not explain some technicalities that are already described in the
previous tutorials, and will only briefly explain how to run deep RL experiments.
Be sure to read the previous tutorials before starting this one.

Solving Atari with DQN
----------------------
This script runs the experiment to solve the Atari Breakout game as described in
the DQN paper *"Human-level control through deep reinforcement learning", Mnih V. et
al., 2015*). We start creating the neural network to learn the action-value
function:

.. literalinclude:: code/dqn.py
   :lines: 1-54

Note that the forward function may return all the action-values of ``state``,
or only the one for the provided ``action``. This network will be used later in
the script.
Now, we define useful functions, set some hyperparameters, and create the ``mdp``
and the policy ``pi``:

.. literalinclude:: code/dqn.py
   :lines: 57-99

Differently from the literature, we use ``Adam`` as the optimizer.

Then, the ``approximator``:

.. literalinclude:: code/dqn.py
   :lines: 101-113

Finally, the ``agent`` and the ``core``:

.. literalinclude:: code/dqn.py
   :lines: 115-129

Eventually, the learning loop is performed. As done in literature, learning and
evaluation steps are alternated:

.. literalinclude:: code/dqn.py
   :lines: 131-158

Solving MuJoCo with DDPG
------------------------
This script runs the experiment to solve the Walker-Stand MuJoCo task, as
implemented in `MuJoCo <https://github.com/deepmind/dm_control/>`_. As with ``DQN``,
we start creating the neural networks. For ``DDPG``, we need an actor and a critic
network:

.. literalinclude:: code/ddpg.py
   :lines: 1-65

We create the ``mdp``, the policy, and set some hyperparameters:

.. literalinclude:: code/ddpg.py
   :lines: 68-83

Note that the policy is not instatiated in the script, since in DDPG the
instatiation is done inside the algorithm constructor.

We create the actor and the critic approximators:

.. literalinclude:: code/ddpg.py
   :lines: 85-102

Finally, we create the ``agent`` and the ``core``:

.. literalinclude:: code/ddpg.py
   :lines: 104-110

As in ``DQN``, we alternate learning and evaluation steps:

.. literalinclude:: code/ddpg.py
   :lines: 112-129
