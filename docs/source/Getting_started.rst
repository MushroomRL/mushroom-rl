.. Mushroom documentation master file, created by
   sphinx-quickstart on Wed Dec  6 10:51:04 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.










===============      
Getting Started
===============
 
















What is MushroomRL
------------------

.. highlight:: python

MushroomRL is a Reinforcement Learning (RL) library developed to be a simple, yet
powerful way to make **RL** and **deep RL** experiments. The idea behind MushroomRL
is to offer the majority of RL algorithms providing a common interface
in order to run them without excessive effort. Moreover, it is designed in such
a way that new algorithms and other stuff can be added transparently,
without the need of editing other parts of the code. MushroomRL is compatible with RL
libraries like   
`OpenAI Gym <https://gym.openai.com/>`_,
`DeepMind Control Suite <https://github.com/deepmind/dm_control>`_,
`Pybullet <https://pybullet.org/wordpress/>`_, and
`MuJoCo <http://www.mujoco.org/>`_, and
the `PyTorch <https://pytorch.org>`_ library for tensor computation.





With MushroomRL you can:
------------------------

- solve RL problems simply writing a single small script  ;  
- add custom algorithms, policies, and so on, transparently;
- use all RL environments offered by well-known libraries and build customized
  environments as well;
- exploit regression models offered by third-party libraries (e.g., scikit-learn) or
  build a customized one with PyTorch;
- seamlessly run experiments on CPU or GPU. 







MushroomRL vs other libraries
-----------------------------
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
   Benchmarking suite              .. centered:: |check|     .. centered:: |check|          .. centered:: |check|     .. centered:: |check|  .. centered:: |check|    .. centered:: |check|
   MujoCo integration              .. centered:: |check|     .. centered:: |cross|          .. centered:: |cross|     .. centered:: |cross|  .. centered:: |cross|    .. centered:: |cross|
   Pybullet integration            .. centered:: |check|     .. centered:: |cross|          .. centered:: |cross|     .. centered:: |cross|  .. centered:: |cross|    .. centered:: |cross|
   Torch integration               .. centered:: |check|     .. centered:: |cross|          .. centered:: |check|     .. centered:: |check|  .. centered:: |cross|    .. centered:: |cross|
   Tensorflow integration          .. centered:: |cross|     .. centered:: |check|          .. centered:: |check|     .. centered:: |check|  .. centered:: |cross|    .. centered:: |check|
   Chainer integration             .. centered:: |cross|     .. centered:: |cross|          .. centered:: |cross|     .. centered:: |cross|  .. centered:: |check|    .. centered:: |cross|
   Parallel environments           .. centered:: |cross|     .. centered:: |check|          .. centered:: |check|     .. centered:: |cross|  .. centered:: |check|    .. centered:: |check|
   ============================== ========================= =============================== ========================= ====================== ======================== =========================











 
