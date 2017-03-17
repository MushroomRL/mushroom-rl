PyPi
******

**PyPi is a toolkit for Reinforcement Learning experiments**

.. contents:: **Contents of this document**
   :depth: 2

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