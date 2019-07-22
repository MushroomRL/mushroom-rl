Mushroom
********

**Mushroom: Reinforcement Learning python library.**

.. contents:: **Contents of this document:**
   :depth: 2

What is Mushroom
================
Mushroom is a python Reinforcement Learning (RL) library whose modularity allows to easily use
well known python libraries for tensor computation (e.g. PyTorch, Tensorflow) and RL benchmark
(e.g. OpenAI Gym, PyBullet). It allows to perform RL experiments in a simple way
providing online TD (e.g. Q-Learning, SARSA), batch TD (e.g. FQI) algorithms, deep RL
algorithms (e.g. DQN and DDPG), and several policy-search algorithms (e.g. REINFORCE, REPS).

Full documentation available at http://mushroomrl.readthedocs.io/en/latest/.

Installation
============

You can do a minimal installation of ``Mushroom`` with:

.. code:: shell

	git clone https://github.com/carloderamo/mushroom.git
	cd mushroom
	pip3 install -e .

Installing everything
---------------------
To install the whole set of features, you will need additional packages installed.
You can install everything by running:

.. code:: shell

	pip3 install -e '.[all]'

This will install every dependency of mushroom, except MuJoCo dependencies.
To use the mujoco-py mushroom interface you can run the command:

.. code:: shell

	pip3 install -e '.[mujoco]'

You might need to install external dependencies first. For more information about mujoco-py
installation follow the instructions on the `project page <https://github.com/openai/mujoco-py>`_

To use dm_control mushroom interface, install dm_control following the instruction that can
be found `here <https://github.com/deepmind/dm_control>`_

How to set and run and experiment
=================================
To run experiments, Mushroom requires a script file that provides the necessary information
for the experiment. Follow the scripts in the "examples" folder to have an idea
of how an experiment can be run.

For instance, to run a quick experiment with one of the provided example scripts, run:

.. code:: shell

    python3 examples/car_on_hill_fqi.py
