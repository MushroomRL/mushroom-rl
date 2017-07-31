PyPi
******

**PyPi: a toolkit for Reinforcement Learning experiments.**

.. contents:: **Contents of this document:**
   :depth: 2

What is PyPi
============
PyPi is a toolkit for solving Reinforcement Learning problems. It is written in Python
and use well-known Machine Learning libraries as Keras (https://keras.io/), Tensorflow (https://www.tensorflow.org/),
Scikit-Learn (http://scikit-learn.org/stable/) and OpenAI Gym (https://gym.openai.com/) libraries.

PyPi is still under development, but it already features the main algorithms used in the
value-based approach in RL (e.g. Q-Learning, SARSA, DQN and others).
By choice, it is currently focused on value-based algorithms, but policy gradient
and actor-critic algorithms will be added in future versions.

Installation
============

You can do a minimal installation of ``PyPi`` with:

.. code:: shell

	git clone https://github.com/carloderamo/PyPi.git
	cd PyPi
	pip install -e .

Installing everything
---------------------
To install the whole set of features, you will need additional packages installed.
You can install everything by running ``pip install -e '.[all]'``.

What's new
==========
- 15-07-2017: New algorithms and environment. DQN and Atari added.
- 17-03-2017: Environments and algorithms added.
- 25-02-2017: Initial release.

How to set and run and experiment
=================================
To run experiments, PyPi requires a script file that provides the necessary information
for the experiment. Follow the scripts in the "examples" folder to have an idea
of how an experiment can be run.

For instance, to run a quick experiment with one of the provided example scripts, run:

.. code:: shell

   python examples/car_on_hill.py