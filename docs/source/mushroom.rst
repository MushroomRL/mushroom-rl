What is MushroomRL
==================

.. highlight:: python

MushroomRL is a Reinforcement Learning (RL) library developed to be a simple, yet
powerful way to make **RL** and **deep RL** experiments. The idea behind MushroomRL
is to offer the majority of RL algorithms providing a common interface
in order to run them without excessive effort. Moreover, it is designed in such
a way that new algorithms and other stuff can be added transparently,
without the need of editing other parts of the code. MushroomRL is compatible with RL
libraries like
`Gymnasium <https://gymnasium.farama.org/>`_,
`DeepMind Control Suite <https://github.com/deepmind/dm_control>`_,
`Pybullet <https://pybullet.org/wordpress/>`_, and
`MuJoCo <http://www.mujoco.org/>`_, and
the `PyTorch <https://pytorch.org>`_ library for tensor computation.

With MushroomRL you can:

- solve RL problems simply writing a single small script;
- use classic RL algorithms and deep RL ones from the same library, behind the same interface;
- add custom algorithms, policies, and so on, transparently;
- use all RL environments offered by well-known libraries and build customized
  environments as well;
- run experiments on MuJoCo, PyBullet, Isaac Sim, Gymnasium and the DeepMind Control Suite;
- exploit regression models offered by third-party libraries (e.g., scikit-learn) or
  build a customized one with PyTorch;
- collect samples from parallel and vectorized environments;
- seamlessly run experiments on CPU or GPU.

Basic run example
-----------------
Solve a discrete MDP in few a lines. Firstly, create a **MDP**:

.. literalinclude:: /source/tutorials/code/basic_run.py
   :lines: 1-3

Then, an epsilon-greedy **policy** with:

.. literalinclude:: /source/tutorials/code/basic_run.py
   :lines: 5-9

Eventually, the **agent** is:

.. literalinclude:: /source/tutorials/code/basic_run.py
   :lines: 11-14

Learn:

.. literalinclude:: /source/tutorials/code/basic_run.py
   :lines: 16-19

Print final Q-table:

.. literalinclude:: /source/tutorials/code/basic_run.py
   :lines: 21-30

Results in:

::

    [[0.6561 0.729  0.6561 0.729 ]
     [0.729  0.81   0.6561 0.81  ]
     [0.81   0.9    0.729  0.81  ]
     [0.6561 0.81   0.729  0.81  ]
     [0.729  0.9    0.729  0.9   ]
     [0.81   1.     0.81   0.9   ]
     [0.729  0.81   0.81   0.9   ]
     [0.81   0.9    0.81   1.    ]
     [0.     0.     0.     0.    ]]

where the Q-values of each action of the MDP are stored for each rows
representing a state of the MDP. 
