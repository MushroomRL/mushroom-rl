MushroomRL
**********

.. image:: https://travis-ci.org/MushroomRL/mushroom-rl.svg?branch=master
   :target: https://travis-ci.org/MushroomRL/mushroom-rl

.. image:: https://readthedocs.org/projects/mushroomrl/badge/?version=latest
   :target: https://mushroomrl.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
    
.. image:: https://api.codeclimate.com/v1/badges/3b0e7167358a661ed882/maintainability
   :target: https://codeclimate.com/github/MushroomRL/mushroom-rl/maintainability
   :alt: Maintainability
   
.. image:: https://api.codeclimate.com/v1/badges/3b0e7167358a661ed882/test_coverage
   :target: https://codeclimate.com/github/MushroomRL/mushroom-rl/test_coverage
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

    pip3 install mushroom_rl[all]

This will install every dependency of MushroomRL, except MuJoCo and Plots dependencies.
For ubuntu>20.04, you may need to install pygame and gym dependencies:

.. code:: shell

    sudo apt -y install libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev \
                     libsdl1.2-dev libsmpeg-dev libportmidi-dev ffmpeg libswscale-dev \
                     libavformat-dev libavcodec-dev swig

To use the ``mujoco-py`` MushroomRL interface you can run the command:

.. code:: shell

    pip3 install mushroom_rl[mujoco]

Below is the code that you need to run to install the Plots dependencies:

.. code:: shell

    sudo apt -y install python3-pyqt5
    pip3 install mushroom_rl[plots]

You might need to install external dependencies first. For more information about mujoco-py
installation follow the instructions on the `project page <https://github.com/openai/mujoco-py>`_

To use dm_control MushroomRL interface, install ``dm_control`` following the instruction that can
be found `here <https://github.com/deepmind/dm_control>`_

You can also perform a local editable installation by using:

.. code:: shell

    pip install --no-use-pep517 -e .


How to set and run and experiment
=================================
To run experiments, MushroomRL requires a script file that provides the necessary information
for the experiment. Follow the scripts in the "examples" folder to have an idea
of how an experiment can be run.

For instance, to run a quick experiment with one of the provided example scripts, run:

.. code:: shell

    python3 examples/car_on_hill_fqi.py
   
Cite Mushroom
=============
If you are using mushroom for your scientific publications, please cite:

.. code:: bibtex

   @misc{deramo2020mushroomrl,
         title={MushroomRL: Simplifying Reinforcement Learning Research},
         author={D'Eramo, Carlo and Tateo, Davide and Bonarini, Andrea and Restelli, Marcello and Peters, Jan},
         journal={arXiv preprint arXiv:2001.01102},
         year={2020},
         howpublished={\url{https://github.com/MushroomRL/mushroom-rl}}
   }

How to contact us
=================
For any question, drop an e-mail at mushroom4rl@gmail.com.

Follow us on Twitter *@Mushroom_RL*!
