Mushroom
********

.. image:: https://travis-ci.org/AIRLab-POLIMI/mushroom.svg?branch=master
    :target: https://travis-ci.org/AIRLab-POLIMI/mushroom
    
.. image:: https://api.codeclimate.com/v1/badges/4a56cb5f751e762bea69/maintainability
   :target: https://codeclimate.com/github/AIRLab-POLIMI/mushroom/maintainability
   :alt: Maintainability
   
.. image:: https://api.codeclimate.com/v1/badges/4a56cb5f751e762bea69/test_coverage
   :target: https://codeclimate.com/github/AIRLab-POLIMI/mushroom/test_coverage
   :alt: Test Coverage

**Mushroom: Reinforcement Learning Python library.**

.. contents:: **Contents of this document:**
   :depth: 2

What is Mushroom
================
Mushroom is a Python Reinforcement Learning (RL) library whose modularity allows
to easily use well-known Python libraries for tensor computation (e.g. PyTorch,
Tensorflow) and RL benchmarks (e.g. OpenAI Gym, PyBullet, Deepmind Control Suite).
It allows to perform RL experiments in a simple way providing classical RL algorithms
(e.g. Q-Learning, SARSA, FQI), and deep RL algorithms (e.g. DQN, DDPG, SAC, TD3,
TRPO, PPO).

Full documentation available at http://mushroomrl.readthedocs.io/en/latest/.

Installation
============

You can do a minimal installation of ``Mushroom`` with:

.. code:: shell

	git clone https://github.com/AIRLab-POLIMI/mushroom.git
	cd mushroom
	pip3 install -e .

Installing everything
---------------------
To install the whole set of features, you will need additional packages installed.
You can install everything by running:

.. code:: shell

	pip3 install -e '.[all]'

This will install every dependency of mushroom, except MuJoCo dependencies.
To use the ``mujoco-py`` mushroom interface you can run the command:

.. code:: shell

	pip3 install -e '.[mujoco]'

You might need to install external dependencies first. For more information about mujoco-py
installation follow the instructions on the `project page <https://github.com/openai/mujoco-py>`_

To use dm_control mushroom interface, install ``dm_control`` following the instruction that can
be found `here <https://github.com/deepmind/dm_control>`_

How to set and run and experiment
=================================
To run experiments, Mushroom requires a script file that provides the necessary information
for the experiment. Follow the scripts in the "examples" folder to have an idea
of how an experiment can be run.

For instance, to run a quick experiment with one of the provided example scripts, run:

.. code:: shell

    python3 examples/car_on_hill_fqi.py
