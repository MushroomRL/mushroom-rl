.. Mushroom documentation master file, created by
   sphinx-quickstart on Wed Dec  6 10:51:04 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==========
MushroomRL
==========

Reinforcement Learning python library
-------------------------------------

.. highlight:: python

MushroomRL is a Reinforcement Learning (RL) library that aims to be a simple, yet
powerful way to make **RL** and **deep RL** experiments. The idea behind MushroomRL
is to offer the majority of RL algorithms providing a common interface
in order to run them without excessive effort. Moreover, it is designed in such
a way that new algorithms and other stuff can be added transparently,
without the need of editing other parts of the code. MushroomRL is compatible with RL
libraries like
`OpenAI Gym <https://gym.openai.com/>`_,
`DeepMind Control Suite <https://github.com/deepmind/dm_control>`_ and
`MuJoCo <http://www.mujoco.org/>`_, and
the `PyTorch <https://pytorch.org>`_ and `Tensorflow <https://www.tensorflow.org/>`_
libraries for tensor computation.

With MushroomRL you can:

- solve RL problems simply writing a single small script;
- add custom algorithms and other stuff transparently;
- use all RL environments offered by well-known libraries and build customized
  environments as well;
- exploit regression models offered by Scikit-Learn or build a customized one
  with PyTorch or Tensorflow;
- seamlessly run experiments on CPU or GPU.

Basic run example
-----------------
Solve a discrete MDP in few a lines. Firstly, create a **MDP**:

::

    from mushroom_rl.environments import GridWorld

    mdp = GridWorld(width=3, height=3, goal=(2, 2), start=(0, 0))

Then, an epsilon-greedy **policy** with:

::

    from mushroom_rl.policy import EpsGreedy
    from mushroom_rl.utils.parameters import Parameter

    epsilon = Parameter(value=1.)
    policy = EpsGreedy(epsilon=epsilon)
                                
Eventually, the **agent** is:

::

    from mushroom_rl.algorithms.value import QLearning

    learning_rate = Parameter(value=.6)
    agent = QLearning(mdp.info, policy, learning_rate)

Learn: 

::

    from mushroom_rl.core.core import Core

    core = Core(agent, mdp)
    core.learn(n_steps=10000, n_steps_per_fit=1)

Print final Q-table:

::

    import numpy as np

    shape = agent.Q.shape
    q = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            state = np.array([i])
            action = np.array([j])
            q[i, j] = agent.Q.predict(state, action)
    print(q)


Results in:

::

    [[  6.561   7.29    6.561   7.29 ]
     [  7.29    8.1     6.561   8.1  ]
     [  8.1     9.      7.29    8.1  ]
     [  6.561   8.1     7.29    8.1  ]
     [  7.29    9.      7.29    9.   ]
     [  8.1    10.      8.1     9.   ]
     [  7.29    8.1     8.1     9.   ]
     [  8.1     9.      8.1    10.   ]
     [  0.      0.      0.      0.   ]]

where the Q-values of each action of the MDP are stored for each rows
representing a state of the MDP.


Download and installation
-------------------------

MushroomRL can be downloaded from the
`GitHub <https://github.com/MushroomRL/mushroom-rl>`_ repository.
Installation can be done running

::

    pip3 install mushroom_rl

To compile the documentation:

::

    cd mushroom_rl/docs
    make html

or to compile the pdf version:

::

    cd mushroom_rl/docs
    make latexpdf

To launch MushroomRL test suite:

::

    pytest

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API:
   :glob:

   source/*

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials:
   :glob:

   source/tutorials/*
