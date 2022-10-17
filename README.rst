**********
MushroomRL
**********

.. image:: https://github.com/MushroomRL/mushroom-rl/actions/workflows/continuous_integration.yml/badge.svg?branch=dev
   :target: https://github.com/MushroomRL/mushroom-rl/actions/workflows/continuous_integration.yml
   :alt: Continuous Integration

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

`Full documentation and tutorials available here <http://mushroomrl.readthedocs.io/en/latest/>`_.

Installation
============

You can do a minimal installation of ``MushroomRL`` with:

.. code:: shell

    pip3 install mushroom_rl

Installing everything
---------------------
``MushroomRL`` contains also some optional components e.g., support for ``OpenAI Gym`` 
environments, Atari 2600 games from the ``Arcade Learning Environment``, and the support
for physics simulators such as ``Pybullet`` and ``MuJoCo``. 
Support for these classes is not enabled by default.

To install the whole set of features, you will need additional packages installed.
You can install everything by running:

.. code:: shell

    pip3 install mushroom_rl[all]

This will install every dependency of MushroomRL, except the Plots dependency.
For ubuntu>20.04, you may need to install pygame and gym dependencies:

.. code:: shell

    sudo apt -y install libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev \
                     libsdl1.2-dev libsmpeg-dev libportmidi-dev ffmpeg libswscale-dev \
                     libavformat-dev libavcodec-dev swig

Notice that you still need to install some of these dependencies for different operating systems, e.g. swig for macOS 

Below is the code that you need to run to install the Plots dependencies:

.. code:: shell

    sudo apt -y install python3-pyqt5
    pip3 install mushroom_rl[plots]

You might need to install external dependencies first. For more information about mujoco-py
installation follow the instructions on the `project page <https://github.com/openai/mujoco-py>`_

    WARNING! when using conda, there may be issues with QT. You can fix them by adding the following lines to the code, replacing ``<conda_base_path>`` with the path to your conda distribution and ``<env_name>`` with the name of the conda environment you are using:
   
.. code:: python

   import os
   os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '<conda_base_path>/envs/<env_name>/bin/platforms'

To use dm_control MushroomRL interface, install ``dm_control`` following the instruction that can
be found `here <https://github.com/deepmind/dm_control>`_

Using Habitat and iGibson with MushroomRL
-----------------------------------------

`Habitat <https://aihabitat.org/>`__ and `iGibson <http://svl.stanford.edu/igibson/>`__
are simulation platforms providing realistic and sensory-rich learning environments.
In MushroomRL, the agent's default observations are RGB images, but RGBD,
agent sensory data, and other information can also be used.

    If you have previous versions of iGibson or Habitat already installed, we recommend to remove them and do clean installs.

iGibson Installation
^^^^^^^^^^^^^^^^^^^^
Follow the `official guide <http://svl.stanford.edu/igibson/#install_env>`__ and install its
`assets <http://svl.stanford.edu/igibson/docs/assets.html>`__ and
`datasets <http://svl.stanford.edu/igibson/docs/dataset.html>`__.

For ``<MUSHROOM_RL PATH>/mushroom-rl/examples/igibson_dqn.py`` you need to run

.. code:: shell

    python -m igibson.utils.assets_utils --download_assets
    python -m igibson.utils.assets_utils --download_demo_data
    python -m igibson.utils.assets_utils --download_ig_dataset

You can also use `third party datasets <https://github.com/StanfordVL/iGibson/tree/master/igibson/utils/data_utils/ext_scene>`__.

The scene details are defined in a YAML file, that needs to be passed to the agent.
See ``<IGIBSON PATH>/igibson/test/test_house.YAML`` for an example.


Habitat Installation
^^^^^^^^^^^^^^^^^^^^
Follow the `official guide <https://github.com/facebookresearch/habitat-lab/#installation>`__
and do a **full install** with `habitat_baselines`.
Then you can download interactive datasets following
`this <https://github.com/facebookresearch/habitat-lab#data>`__ and
`this <https://github.com/facebookresearch/habitat-lab#task-datasets>`__.
If you need to download other datasets, you can use
`this utility <https://github.com/facebookresearch/habitat-sim/blob/master/habitat_sim/utils/datasets_download.py>`__.

Basic Usage of Habitat
^^^^^^^^^^^^^^^^^^^^^^
When you create a ``Habitat`` environment, you need to pass a wrapper name and two
YAML files: ``Habitat(wrapper, config_file, base_config_file)``.

* The wrapper has to be among the ones defined in ``<MUSHROOM_RL PATH>/mushroom-rl/environments/habitat_env.py``,
  and takes care of converting actions and observations in a gym-like format. If your task / robot requires it,
  you may need to define new wrappers.

* The YAML files define every detail: the Habitat environment, the scene, the
  sensors available to the robot, the rewards, the action discretization, and any
  additional information you may need. The second YAML file is optional, and
  overwrites whatever was already defined in the first YAML.

    If you use YAMLs from ``habitat-lab``, check if they define a YAML for
    ``BASE_TASK_CONFIG_PATH``. If they do, you need to pass it as ``base_config_file`` to
    ``Habitat()``. ``habitat-lab`` YAMLs, in fact, use relative paths, and calling them
    from outside its root folder will cause errors.

* If you use a dataset, be sure that the path defined in the YAML file is correct,
  especially if you use relative paths. ``habitat-lab`` YAMLs use relative paths, so
  be careful with that. By default, the path defined in the YAML file will be
  relative to where you launched the python code. If your `data` folder is
  somewhere else, you may also create a symbolic link.

Rearrange Task Example
^^^^^^^^^^^^^^^^^^^^^^
* Download the ReplicaCAD datasets (``--data-path data`` downloads them in the folder
  from where you are launching your code)

.. code:: shell

    python -m habitat_sim.utils.datasets_download --uids replica_cad_dataset --data-path data

* For this task we use ``<HABITAT_LAB PATH>/habitat_baselines/config/rearrange/rl_pick.yaml``.
  This YAML defines ``BASE_TASK_CONFIG_PATH: configs/tasks/rearrange/pick.yaml``,
  and since this is a relative path we need to overwrite it by passing its absolute path
  as ``base_config_file`` argument to ``Habitat()``.

* Then, ``pick.yaml`` defines the dataset to be used with respect to ``<HABITAT_LAB PATH>``.
  If you have not used ``--data-path`` argument with the previous download command,
  the ReplicaCAD datasets is now in ``<HABITAT_LAB PATH>/data`` and you need to
  make a link to it

.. code:: shell

    ln -s <HABITAT_LAB PATH>/data/ <MUSHROOM_RL PATH>/mushroom-rl/examples/habitat

* Finally, you can launch ``python habitat_rearrange_sac.py``.

Navigation Task Example
^^^^^^^^^^^^^^^^^^^^^^^
* Download and extract Replica scenes

    WARNING! The dataset is very large!

.. code:: shell

    sudo apt-get install pigz
    git clone https://github.com/facebookresearch/Replica-Dataset.git
    cd Replica-Dataset
    ./download.sh replica-path

* For this task we only use the custom YAML file ``pointnav_apartment-0.yaml``.

* ``DATA_PATH: "replica_{split}_apartment-0.json.gz"`` defines the JSON file with
  some scene details, such as the agent's initial position and orientation.
  The ``{split}`` value is defined in the ``SPLIT`` key.

    If you want to try new positions, you can sample some from the set of the scene's navigable points.
    After initializing a ``habitat`` environment, for example ``mdp = Habitat(...)``,
    run ``mdp.env._env._sim.sample_navigable_point()``.

* ``SCENES_DIR: "Replica-Dataset/replica-path/apartment_0"`` defines the scene.
  As said before, this path is relative to where you launch the script, thus we need to link the Replica folder.
  If you launch ``habitat_nav_dqn.py`` from its example folder, run

.. code:: shell

    ln -s <PATH TO>/Replica-Dataset/ <MUSHROOM_RL PATH>/mushroom-rl/examples/habitat

* Finally, you can launch ``python habitat_nav_dqn.py``.



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
