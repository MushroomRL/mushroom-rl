Download and installation
=========================

MushroomRL requires Python 3.11 or later.

Installing from PyPI
--------------------

To use MushroomRL as a library, install the released package:

::

    pip install mushroom-rl

Installing from source
----------------------

To follow the development version, or to modify MushroomRL itself, clone the
`GitHub <https://github.com/MushroomRL/mushroom-rl>`_ repository and install it in editable mode, so that your
changes take effect without reinstalling:

::

    git clone https://github.com/MushroomRL/mushroom-rl.git
    cd mushroom-rl
    pip install -e .

Optional dependencies
---------------------

The environments built on an external simulator are installed through extras. Each one is optional: the
corresponding environments register themselves only if their dependency imports successfully, so an environment
missing from ``Environment.list_registered()`` means its extra is not installed.

.. csv-table::
   :header: "Extra", "What it enables"
   :widths: 24, 76

   "``gymnasium``", "The Gymnasium environments"
   "``atari``", "The Arcade Learning Environment games"
   "``minigrid``", "The MiniGrid grid worlds"
   "``mujoco``", "The MuJoCo environments"
   "``dm_control``", "The DeepMind Control Suite"
   "``bullet``", "The PyBullet environments"
   "``box2d``", "The Box2D Gymnasium environments"
   "``monitors``", "The live plotting windows (:doc:`api/utils/monitors`)"
   "``wandb``", "Logging to Weights & Biases"
   "``all``", "Everything above **except** ``box2d`` and ``bullet``"

Install one or more of them with the usual syntax, e.g.

::

    pip install mushroom-rl[mujoco,wandb]

or, from a clone,

::

    pip install -e .[all]

The Isaac Sim environments are not covered by an extra: Isaac Sim has to be installed separately, following
NVIDIA's instructions.

Building the documentation
--------------------------

To compile this documentation:

::

    cd mushroom-rl/docs
    make html

or, for the pdf version:

::

    cd mushroom-rl/docs
    make latexpdf

Running the test suite
----------------------

From the repository root:

::

    pytest

Installation troubleshooting
----------------------------
Common problems with the installation of MushroomRL arise in case some of its dependencies are
broken or not installed. In general, we recommend installing MushroomRL with the option ``all`` to install all the Python
dependencies. The installation time mostly depends on the time to install the dependencies.
A simple installation takes approximately 1 minute with a fast internet connection.
Installing with all the dependencies takes approximately 5 minutes using a fast internet connection. A slower
internet connection may increase the installation time significantly.

If installing all the dependencies, ensure that the SWIG library is installed, as it is used
by some Gymnasium environments and the installation may fail otherwise. For Atari, you might need to install the ROMs
separately, otherwise the creation of Atari environments may fail. OpenCV should be installed too.
Installing MushroomRL in a Conda environment is generally safe.

To check if the installation has been successful, try to run the basic example.

MushroomRL is well-tested on Linux. If you are using another OS, you may run into issues that
we are still not aware of. In that case, please do not hesitate to send us an email at mushroom4rl@gmail.com.
