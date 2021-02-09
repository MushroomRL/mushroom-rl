How to make a simple experiment
===============================

The main purpose of MushroomRL is to simplify the scripting of RL experiments. A
standard example of a script to run an experiment in MushroomRL, consists of:

* an **initial part** where the setting of the experiment are specified;
* a **middle part** where the experiment is run;
* a **final part** where operations like evaluation, plot and save can be done.

A RL experiment consists of:

* a **MDP**;
* an **agent**;
* a **core**.

A **MDP** is the problem to be solved by the agent. It contains the function to move
the agent in the environment according to the provided action.
The MDP can be simply created with:

.. literalinclude:: code/simple_experiment.py
   :lines: 1-11

A MushroomRL **agent** is the algorithm that is run to learn in the MDP. It consists
of a policy approximator and of the methods to improve the policy during the
learning. It also contains the features to extract in the case of MDP with continuous
state and action spaces. An agent can be defined this way:

.. literalinclude:: code/simple_experiment.py
   :lines: 13-27

This piece of code creates the policy followed by the agent (e.g. :math:`\varepsilon`-greedy)
with :math:`\varepsilon = 1`. Then, the policy approximator is created specifying the
parameters to create it and the class (in this case, the ``ExtraTreesRegressor`` class
of scikit-learn is used). Eventually, the agent is created calling the algorithm
class and providing the approximator and the policy, together with parameters used
by the algorithm.

To run the experiment, the **core** module has to be used. This module requires
the agent and the MDP object and contains the function to learn in the MDP and
evaluate the learned policy. It can be created with:

.. literalinclude:: code/simple_experiment.py
   :lines: 29

Once the core has been created, the agent can be trained collecting a dataset and
fitting the policy:

.. literalinclude:: code/simple_experiment.py
   :lines: 31

In this case, the agent's policy is fitted only once, after that 1000 episodes
have been collected. This is a common practice in batch RL algorithms such as
``FQI`` where, initially, samples are randomly collected and then the policy is fitted
using the whole dataset of collected samples.

Eventually, some operations to evaluate the learned policy can be done.
This way the user can, for instance, compute the performance of the agent
through the collected rewards during an evaluation run.
Fixing :math:`\varepsilon = 0`, the greedy policy is applied starting from the
provided initial states, then the average cumulative discounted reward is returned.

.. literalinclude:: code/simple_experiment.py
   :lines: 33-
