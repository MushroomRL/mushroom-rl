Mushroom
******

**Mushroom: Reinforcement Learning python library.**

.. contents:: **Contents of this document:**
   :depth: 2

What is Mushroom
============
Mushroom is a python Reinforcement Learning (RL) library using Tensorflow and
OpenAI Gym libraries. It allows to perform RL in a simple way providing TD (e.g. Q-Learning, SARSA)
and batch TD (e.g. FQI) algorithms, together with the famous DQN algorithm used to solve the Atari environment.

By choice, it is currently focused on value-based algorithms, but policy gradient
and actor-critic algorithms will be added in future versions.

Full documentation available at https://readthedocs.org/projects/mushroomrl/.

Installation
============

You can do a minimal installation of ``Mushroom`` with:

.. code:: shell

	git clone https://github.com/carloderamo/mushroom.git
	cd mushroom
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