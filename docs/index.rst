
:html_theme.sidebar_secondary.remove:



.. Mushroom documentation master file, created by
   sphinx-quickstart on Wed Dec  6 10:51:04 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



.. image:: _static/output_image.jpeg
  :width: 500
  :align: center
  :class: only-light


.. image:: _static/title_dark.jpeg
  :width: 500
  :align: center
  :class: only-dark







A Library for Reinforcement learning
-------------------------------------
 





|pic2| |pic9|
   

.. |pic2| image:: _static/walker_walk_sac.gif
   :width: 420
   :height: 15em
   




.. |pic9| image:: _static/defend.gif
   :width: 420
   :height: 15em
   




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


 
.. image:: _static/breakout_dqn.gif
  :width: 400  
  :height: 15em
  :align: center  







.. toctree::
   :caption: Getting Started:
   :maxdepth: 1
   :glob:
   :hidden:

   source/user_guide


.. toctree::
   :caption: API:
   :maxdepth: 2
   :glob:
   :hidden:

   /Features



.. toctree::
   :caption: Tutorials:
   :maxdepth: 2
   :glob:
   :hidden:

   /Tutorials





