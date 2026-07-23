.. Mushroom documentation master file, created by
   sphinx-quickstart on Wed Dec  6 10:51:04 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==========
MushroomRL
==========

What is MushroomRL
==================

.. highlight:: python

MushroomRL is a Reinforcement Learning (RL) library developed to be a simple, yet
powerful way to make **RL** and **deep RL** experiments. The idea behind MushroomRL
is to offer the majority of RL algorithms providing a common interface
in order to run them without excessive effort. Moreover, it is designed in such
a way that new algorithms and other stuff can be added transparently,
without the need of editing other parts of the code. MushroomRL is compatible with RL
libraries like
`Gymnasium <https://gymnasium.farama.org/>`_,
`DeepMind Control Suite <https://github.com/deepmind/dm_control>`_,
`Pybullet <https://pybullet.org/wordpress/>`_, and
`MuJoCo <http://www.mujoco.org/>`_, and
the `PyTorch <https://pytorch.org>`_ library for tensor computation.

With MushroomRL you can:

- solve RL problems simply writing a single small script;
- add custom algorithms, policies, and so on, transparently;
- use all RL environments offered by well-known libraries and build customized
  environments as well;
- exploit regression models offered by third-party libraries (e.g., scikit-learn) or
  build a customized one with PyTorch;
- seamlessly run experiments on CPU or GPU.

Basic run example
-----------------
Solve a discrete MDP in few a lines. Firstly, create a **MDP**:

.. literalinclude:: source/tutorials/code/basic_run.py
   :lines: 1-3

Then, an epsilon-greedy **policy** with:

.. literalinclude:: source/tutorials/code/basic_run.py
   :lines: 5-9

Eventually, the **agent** is:

.. literalinclude:: source/tutorials/code/basic_run.py
   :lines: 11-14

Learn:

.. literalinclude:: source/tutorials/code/basic_run.py
   :lines: 16-19

Print final Q-table:

.. literalinclude:: source/tutorials/code/basic_run.py
   :lines: 21-30

Results in:

::

    [[0.6561 0.729  0.6561 0.729 ]
     [0.729  0.81   0.6561 0.81  ]
     [0.81   0.9    0.729  0.81  ]
     [0.6561 0.81   0.729  0.81  ]
     [0.729  0.9    0.729  0.9   ]
     [0.81   1.     0.81   0.9   ]
     [0.729  0.81   0.81   0.9   ]
     [0.81   0.9    0.81   1.    ]
     [0.     0.     0.     0.    ]]

where the Q-values of each action of the MDP are stored for each rows
representing a state of the MDP.

Download and installation
=========================

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

To check if the installation has been successful, you can try to run the basic example above.

MushroomRL is well-tested on Linux. If you are using another OS, you may run into issues that
we are still not aware of. In that case, please do not hesitate to send us an email at mushroom4rl@gmail.com.

MushroomRL vs other libraries
=============================
MushroomRL offers the majority of classical and deep RL algorithms, while keeping a modular
and flexible architecture. It is compatible with Pytorch, and most machine learning and RL
libraries.

.. |check| unicode:: U+2705

.. |cross| unicode:: U+274C


.. table::

   ============================== ========================= =============================== ========================= ====================== ======================== =========================
   Features                       .. centered:: MushroomRL  .. centered:: Stable Baselines   .. centered:: RLLib      .. centered:: Keras RL .. centered:: Chainer RL .. centered:: Tensorforce
   ============================== ========================= =============================== ========================= ====================== ======================== =========================
   Classic RL algorithms           .. centered:: |check|     .. centered:: |cross|          .. centered:: |cross|     .. centered:: |cross|  .. centered:: |cross|    .. centered:: |cross|
   Deep RL algorithms              .. centered:: |check|     .. centered:: |check|          .. centered:: |check|     .. centered:: |cross|  .. centered:: |check|    .. centered:: |cross|
   Updated documentation           .. centered:: |check|     .. centered:: |check|          .. centered:: |check|     .. centered:: |cross|  .. centered:: |check|    .. centered:: |check|
   Modular                         .. centered:: |check|     .. centered:: |cross|          .. centered:: |cross|     .. centered:: |cross|  .. centered:: |check|    .. centered:: |check|
   Easy to extend                  .. centered:: |check|     .. centered:: |cross|          .. centered:: |cross|     .. centered:: |cross|  .. centered:: |cross|    .. centered:: |cross|
   PEP8 compliant                  .. centered:: |check|     .. centered:: |check|          .. centered:: |check|     .. centered:: |check|  .. centered:: |check|    .. centered:: |check|
   Compatible with RL benchmarks   .. centered:: |check|     .. centered:: |check|          .. centered:: |check|     .. centered:: |cross|  .. centered:: |check|    .. centered:: |check|
   Parallel environments           .. centered:: |check|     .. centered:: |check|          .. centered:: |check|     .. centered:: |cross|  .. centered:: |check|    .. centered:: |check|
   Benchmarking suite              .. centered:: |check|     .. centered:: |check|          .. centered:: |check|     .. centered:: |check|  .. centered:: |check|    .. centered:: |check|
   MujoCo integration              .. centered:: |check|     .. centered:: |cross|          .. centered:: |cross|     .. centered:: |cross|  .. centered:: |cross|    .. centered:: |cross|
   Pybullet integration            .. centered:: |check|     .. centered:: |cross|          .. centered:: |cross|     .. centered:: |cross|  .. centered:: |cross|    .. centered:: |cross|
   Torch integration               .. centered:: |check|     .. centered:: |cross|          .. centered:: |check|     .. centered:: |check|  .. centered:: |cross|    .. centered:: |cross|
   Tensorflow integration          .. centered:: |cross|     .. centered:: |check|          .. centered:: |check|     .. centered:: |check|  .. centered:: |cross|    .. centered:: |check|
   Chainer integration             .. centered:: |cross|     .. centered:: |cross|          .. centered:: |cross|     .. centered:: |cross|  .. centered:: |check|    .. centered:: |cross|
   ============================== ========================= =============================== ========================= ====================== ======================== =========================

.. toctree::
   :caption: Tutorials
   :maxdepth: 2
   :glob:

   source/tutorials/*


.. toctree::
   :caption: API
   :maxdepth: 2
   :glob:

   source/*



