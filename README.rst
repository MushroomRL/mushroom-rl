Mushroom
******

**Mushroom: a toolkit for Reinforcement Learning experiments.**

.. contents:: **Contents of this document:**
   :depth: 2

What is Mushroom
============
Mushroom is a toolkit for solving Reinforcement Learning problems. It is written in Python
and makes a large use of Tensorflow (https://www.tensorflow.org/) and
OpenAI Gym (https://gym.openai.com/) libraries.

Mushroom is still under development, but it already features the main algorithms used in the
value-based approach in RL (e.g. Q-Learning, SARSA, FQI) and deep RL (e.g. DQN, Double DQN).
By choice, it is currently focused on value-based algorithms, but policy gradient
and actor-critic algorithms will be added in future versions.

Installation
============

You can do a minimal installation of ``Mushroom`` with:

.. code:: shell

	git clone https://github.com/carloderamo/Mushroom.git
	cd Mushroom
	pip install -e .

Installing everything
---------------------
To install the whole set of features, you will need additional packages installed.
You can install everything by running ``pip install -e '.[all]'``.

How to set and run and experiment
=================================
To run experiments, Mushroom requires a script file that provides the necessary information
for the experiment. Follow the scripts in the "examples" folder to have an idea
of how an experiment can be run.

For instance, to run a quick experiment with one of the provided example scripts, run:

.. code:: shell

   python examples/car_on_hill.py