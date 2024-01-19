.. Mushroom documentation master file, created by
   sphinx-quickstart on Wed Dec  6 10:51:04 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



============     
User Guide
============



Download and installation
-------------------------

MushroomRL can be downloaded from the
`GitHub <https://github.com/MushroomRL/mushroom-rl>`_ repository.
Installation can be done running

::

    pip3 install mushroom_rl

To compile the documentation:

::

    cd mushroom_rl/docs
    make html

or to compile the pdf version:

::

    cd mushroom_rl/docs
    make latexpdf

To launch MushroomRL test suite:

::

    pytest

Installation troubleshooting
----------------------------
Common problems with the installation of MushroomRL arise in case some of its dependency are
broken or not installed. In general, we recommend installing MushroomRL with the option ``all`` to install all the Python
dependencies. The installation time mostly depends on the time to install the dependencies.
A simple installation takes approximately 1 minute with a fast internet connection.
Installing with all the dependencies takes approximately 5 minutes using a fast internet connection. A slower
internet connection may increase the installation time significantly.

If installing all the dependencies, ensure that the swig library is installed, as it is used
by some Gym environments and the installation may fail otherwise. For Atari, you might need to install the ROM separately, otherwise
the creation of Atari environments may fail. Opencv should be installed too. For MuJoCo, ensure that the path of your MuJoCo folder is included
in the environment variable ``LD_LIBRARY_PATH`` and that ``mujoco_py`` is correctly installed.
Installing MushroomRL in a Conda environment is generally
safe. However, we are aware that when installing with the option
``plots``, some errors may arise due to incompatibility issues between
``pyqtgraph`` and Conda. We recommend not using Conda when installing using ``plots``.
Finally, ensure that C/C++ compilers and Cython are working as expected.

To check if the installation has been successful, you can try to run the basic example above.

MushroomRL is well-tested on Linux. If you are using another OS, you may incur in issues that
we are still not aware of. In that case, please do not hesitate sending us an email at mushroom4rl@gmail.com.
























