PyPi
******

**PyPi: a toolkit for Reinforcement Learning experiments.**

.. contents:: **Contents of this document:**
   :depth: 2

What is PyPi
============
PyPi is a toolkit for solving Reinforcement Learning problems. It is written in Python
and makes a large use of Scikit-Learn (http://scikit-learn.org/stable/), Keras (https://keras.io/)
and OpenAI Gym (https://gym.openai.com/) libraries. These libraries are well-known in the Machine Learning
community and helps PyPi solving Reinforcement Learning problems significantly
decreasing the implementation complexity.

PyPi has the purpose to help Machine Learning researchers in solving Reinforcement
Learning problems.

PyPi is still under development and has only a small amount
of algorithms and MDPs implemented, but new features will be added soon. By choice,
it is currently focused on value-based algorithms. Policy gradient and actor-critic
algorithms will be added once the value-based section will be completed.

Installation
============

You can perform a minimal install of ``PyPi`` with:

.. code:: shell

	git clone https://github.com/carloderamo/PyPi.git
	cd PyPi
	pip install -e .

Installing everything
---------------------
To install the whole set of features, you will need additional packages installed.
You can install everything by running ``pip install -e '.[all]'``.

What's new
----------
- 17-03-2017: Environments and algorithms added.
- 25-02-2017: Initial release.

How to set and run and experiment
=================================
PyPi requires a configuration .json file where the setting of the experiment is
described. This file should contain the names of the algorithm, the approximator
used to approximate the target function (e.g. Q-function), the environment to
solve and the policy to use. Parameters for each of these objects have to be
specified too.

Configuration files are in the /config folder and have to follow the same
structure of the already present files.

To run a quick experiment with one of the provided example scripts, run:

.. code:: shell

   python examples/td.py --config config/car_on_hill.json