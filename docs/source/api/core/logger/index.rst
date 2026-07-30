Logger
======

.. module:: mushroom_rl.core.logger

The ``Logger`` is the single entry point for the output of an experiment. It combines a ``DataLogger``, which saves
numpy arrays and agents under a run directory, with a ``ConsoleLogger``, which prints progress to the terminal, and
extends both with the metric-logging methods the agents call. Attach it once with ``Core.set_logger`` and it is
forwarded down the object tree, so every component that declares itself loggable reaches the same run.

``VideoLogger`` and ``WandbLogger`` are mixins adding video recording and Weights & Biases reporting to the same
object.

.. autosummary::
   :nosignatures:

   Logger
   ConsoleLogger
   DataLogger
   VideoLogger
   WandbLogger

.. toctree::
   :maxdepth: 1

   logger
   console_logger
   data_logger
   video_logger
   wandb_logger
