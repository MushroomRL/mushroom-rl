MushroomRL
**********

.. image:: https://travis-ci.org/AIRLab-POLIMI/mushroom-rl.svg?branch=master
   :target: https://travis-ci.org/AIRLab-POLIMI/mushroom-rl

.. image:: https://readthedocs.org/projects/mushroomrl/badge/?version=latest
   :target: https://mushroomrl.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
    
.. image:: https://api.codeclimate.com/v1/badges/3de0080368fd3f390a66/maintainability
   :target: https://codeclimate.com/github/AIRLab-POLIMI/mushroom-rl/maintainability
   :alt: Maintainability
   
.. image:: https://api.codeclimate.com/v1/badges/3de0080368fd3f390a66/test_coverage
   :target: https://codeclimate.com/github/AIRLab-POLIMI/mushroom-rl/test_coverage
   :alt: Test Coverage

**MushroomRL: Reinforcement Learning Python library.**

.. contents:: **Contents of this document:**
   :depth: 2

What is MushroomRL
==================
MushroomRL is a Python Reinforcement Learning (RL) library whose modularity allows
to easily use well-known Python libraries for tensor computation (e.g. PyTorch,
Tensorflow) and RL benchmarks (e.g. OpenAI Gym, PyBullet, Deepmind Control Suite).
It allows to perform RL experiments in a simple way providing classical RL algorithms
(e.g. Q-Learning, SARSA, FQI), and deep RL algorithms (e.g. DQN, DDPG, SAC, TD3,
TRPO, PPO).

Full documentation available `here <http://mushroomrl.readthedocs.io/en/latest/>`_.

Installation
============

You can do a minimal installation of ``MushroomRL`` with:

.. code:: shell

	pip3 install mushroom_rl

Installing everything
---------------------
To install the whole set of features, you will need additional packages installed.
You can install everything by running:

.. code:: shell

	pip3 install mushroom_rl '.[all]'

This will install every dependency of MushroomRL, except MuJoCo dependencies.
To use the ``mujoco-py`` MushroomRL interface you can run the command:

.. code:: shell

	pip3 install mushroom_rl '.[mujoco]'

You might need to install external dependencies first. For more information about mujoco-py
installation follow the instructions on the `project page <https://github.com/openai/mujoco-py>`_

To use dm_control MushroomRL interface, install ``dm_control`` following the instruction that can
be found `here <https://github.com/deepmind/dm_control>`_

How to set and run and experiment
=================================
To run experiments, MushroomRL requires a script file that provides the necessary information
for the experiment. Follow the scripts in the "examples" folder to have an idea
of how an experiment can be run.

For instance, to run a quick experiment with one of the provided example scripts, run:

.. code:: shell

    python3 examples/car_on_hill_fqi.py
