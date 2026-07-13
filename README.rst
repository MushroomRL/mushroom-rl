**********
MushroomRL
**********

.. image:: https://github.com/MushroomRL/mushroom-rl/actions/workflows/continuous_integration.yml/badge.svg?branch=dev
   :target: https://github.com/MushroomRL/mushroom-rl/actions/workflows/continuous_integration.yml
   :alt: Continuous Integration

.. image:: https://readthedocs.org/projects/mushroomrl/badge/?version=latest
   :target: https://mushroomrl.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://qlty.sh/gh/MushroomRL/projects/mushroom-rl/maintainability.svg
   :target: https://qlty.sh/gh/MushroomRL/projects/mushroom-rl
   :alt: Maintainability

.. image:: https://qlty.sh/gh/MushroomRL/projects/mushroom-rl/coverage.svg
   :target: https://qlty.sh/gh/MushroomRL/projects/mushroom-rl
   :alt: Test Coverage

**MushroomRL: Reinforcement Learning Python library.**

.. contents:: **Contents of this document:**
   :depth: 2

What is MushroomRL
==================
MushroomRL is a Python Reinforcement Learning (RL) library whose modularity allows
to easily use well-known Python libraries for tensor computation (e.g. PyTorch,
Tensorflow) and RL benchmarks (e.g. Gymnasium, PyBullet, Deepmind Control Suite).
It allows to perform RL experiments in a simple way providing classical RL algorithms
(e.g. Q-Learning, SARSA, FQI), and deep RL algorithms (e.g. DQN, DDPG, SAC, TD3,
TRPO, PPO).

`Full documentation and tutorials available here <http://mushroomrl.readthedocs.io/en/latest/>`_.

Installation
============

You can do a minimal installation of ``MushroomRL`` with:

.. code:: shell

    pip3 install mushroom_rl

Installing everything
---------------------
``MushroomRL`` contains also some optional components e.g., support for ``Gymnasium``
environments, Atari 2600 games from the ``Arcade Learning Environment``, and the support
for physics simulators such as ``Pybullet`` and ``MuJoCo``. 
Support for these classes is not enabled by default.

To install the whole set of features, you will need additional packages installed.
You can install everything by running:

.. code:: shell

    pip3 install mushroom_rl[all]

This will install every dependency of MushroomRL, except Box2D and PyBullet.
For ubuntu>20.04, you may need to install pygame and gym dependencies:

.. code:: shell

    sudo apt -y install libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev \
                     libsdl1.2-dev libsmpeg-dev libportmidi-dev ffmpeg libswscale-dev \
                     libavformat-dev libavcodec-dev swig

Notice that you still need to install some of these dependencies for different operating systems, e.g. swig for macOS 

Below is the code that you need to run to install the Plots dependencies:

.. code:: shell

    pip3 install mushroom_rl[plots]

To use dm_control MushroomRL interface, install ``dm_control`` following the instruction that can
be found `here <https://github.com/deepmind/dm_control>`_

Editable Installation
---------------------

You can also perform a local editable installation by using:

.. code:: shell

    pip install --no-use-pep517 -e .

To install also optional dependencies:

.. code:: shell

    pip install --no-use-pep517 -e .[all]



How to set and run and experiment
=================================
To run experiments, MushroomRL requires a script file that provides the necessary information
for the experiment. Follow the scripts in the "examples" folder to have an idea
of how an experiment can be run.

For instance, to run a quick experiment with one of the provided example scripts, run:

.. code:: shell

    python3 examples/car_on_hill_fqi.py

Cite MushroomRL
===============
If you are using MushroomRL for your scientific publications, please cite:

.. code:: bibtex

    @article{JMLR:v22:18-056,
        author  = {Carlo D'Eramo and Davide Tateo and Andrea Bonarini and Marcello Restelli and Jan Peters},
        title   = {MushroomRL: Simplifying Reinforcement Learning Research},
        journal = {Journal of Machine Learning Research},
        year    = {2021},
        volume  = {22},
        number  = {131},
        pages   = {1-5},
        url     = {http://jmlr.org/papers/v22/18-056.html}
    }

How to contact us
=================
For any question, drop an e-mail at mushroom4rl@gmail.com.

Follow us on Twitter `@Mushroom_RL <https://twitter.com/mushroom_rl>`_!
