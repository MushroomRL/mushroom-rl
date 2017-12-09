.. Mushroom documentation master file, created by
   sphinx-quickstart on Wed Dec  6 10:51:04 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

========
Mushroom
========

Reinforcement Learning python library
-------------------------------------

.. highlight:: python

Mushroom is a Reinforcement Learning (RL) library that aims to be a simple, yet
powerful way to make **RL** and **deep RL** experiments. The idea behind Mushroom
consists in offering the majority of RL algorithms providing a common interface
in order to run them without excessive effort. Moreover, it is designed in such
a way that new algorithms and other stuff can generally be added transparently
without the need of editing other parts of the code. Mushroom makes a large use
of the environments provided by
`OpenAI Gym <https://gym.openai.com/>`_ library and of the regression models
provided by `Scikit-Learn <http://scikit-learn.org/stable/>`_ library giving
also the possibility to build and run neural networks using
`Tensorflow <https://www.tensorflow.org>`_ library.

With Mushroom you can:

- solve RL problems simply writing a single small script;
- add custom algorithms and other stuff transparently;
- use all RL environments offered by OpenAI Gym and build customized
  environments as well;
- exploit regression models offered by Scikit-Learn or build a customized one
  with Tensorflow;
- run experiments with CPU or GPU.

Basic run example
-----------------
Solve a discrete MDP in few a lines. Firstly, create a **MDP**:

::

    from mushroom.environments import GridWorld

    mdp = GridWorld(width=3, height=3, goal=(2, 2), start=(0, 0))

Then, an epsilon-greedy **policy** with:

::

    from mushroom.policy import EpsGreedy
    from mushroom.utils.parameters import Parameter

    epsilon = Parameter(value=1.)
    policy = EpsGreedy(epsilon=epsilon)
                                
Eventually, the **agent** is:

::

    from mushroom.algorithms.value import QLearning

    learning_rate = Parameter(value=.6)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = QLearning(pi, mdp.info, agent_params)

Learn: 

::

    from mushroom.core.core import Core

    core = Core(agent, mdp)
    core.learn(n_steps=10000, n_steps_per_fit=1)

Print final Q-table:

::

    import numpy as np

    shape = agent.approximator.shape
    q = np.zeros(shape)
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            state = np.array([i])
            action = np.array([j])
            q[i, j] = agent.approximator.predict(state, action)
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

Mushroom can be downloaded from the
`GitHub <https://github.com/carloderamo/mushroom>`_ repository.
Installation can be done running

::

    pip install -e .
    
and

::

    pip install -r requirements.txt
    
to install all its dependencies.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   source/mushroom
