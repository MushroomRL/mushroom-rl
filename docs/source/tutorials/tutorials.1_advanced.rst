Make an advanced experiment
==================================

Continuous MDPs are a challenging class of problems to solve in RL. In these
problems, a tabular regressor is not enough to approximate the Q-function, since
there are an infinite number of states/actions. The solution to solve them is to
use a function approximator (e.g. neural network) fed with the raw values
of states and actions. In the case a linear approximator is used, it is
convenient to enlarge the input space with the space of non-linear **features**
extracted from the raw values. This way, the linear approximator is often able
to solve the MDPs, despite its simplicity. Many RL algorithms rely on the use of
a linear approximator to solve a MDP, therefore the use of features is very
important.
This tutorial shows how to solve a continuous MDP in MushroomRL using an
algorithm that requires the use of a linear approximator.

Initially, the MDP and the policy are created:

.. literalinclude:: code/advanced_experiment.py
   :lines: 1-19

This is an environment created with the MushroomRL interface to the OpenAI Gym
library. Each environment offered by OpenAI Gym can be created this way simply
providing the corresponding id in the ``name`` parameter, except for the Atari
that are managed by a separate class.
After the creation of the MDP, the tiles features are created:

.. literalinclude:: code/advanced_experiment.py
   :lines: 21-30

In this example, we use sparse coding by means of **tiles** features. The
``generate`` method generates ``n_tilings`` grids of 10x10 tilings evenly spaced
(the way the tilings are created is explained in *"Reinforcement Learning: An Introduction",
Sutton & Barto, 1998*). Eventually, the grid is passed to the ``Features``
factory method that returns the features class.

MushroomRL offers other type of features such a **radial basis functions** and
**polynomial** features. The former have also a faster implementation written in
Tensorflow that can be used transparently.

Then, the agent is created as usual, but this time passing the feature to it.
It is important to notice that the learning rate is divided by the number of
tilings for the correctness of the update (see *"Reinforcement Learning: An Introduction",
Sutton & Barto, 1998* for details). After that, the learning is run as usual:

.. literalinclude:: code/advanced_experiment.py
   :lines: 32-46

To visualize the learned policy the rendering method of OpenAI Gym is used. To
activate the rendering in the environments that supports it, it is necessary to
set ``render=True``.

.. literalinclude:: code/advanced_experiment.py
   :lines: 48-
