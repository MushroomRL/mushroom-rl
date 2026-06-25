How to use the Logger
=====================

Here we explain in detail the usage of the MushroomRL Logger class.
This class can be used as a standardized console logger and can also log on disk
Numpy arrays or a mushroom agent, using the appropriate logging folder.

Constructing the Logger
-----------------------

To initialize the logger we can simply choose a log directory and an experiment name:

.. literalinclude:: code/logger.py
    :lines: 1-5

This will create the experiment folder named 'tutorial' inside the base folder '/tmp/logs'.
The logger creates all the necessary directories if they do not exist.
If ``results_dir`` is not specified, the log will create a './logs' base directory.
By setting ``log_console`` to true, the logger will store the console output in a '.log' text file inside the experiment folder, with the same name.
If the file already exists, the logger will append the new logged lines.

If you do not want the logger to create any directory e.g., to only use the log for the console
output, you can force the ``results_dir`` parameter to None:

.. literalinclude:: code/logger.py
   :lines: 7-8

Logging message on the console
------------------------------

The most basic functionality of the Logger is to output text messages on the standard output.
Our logger uses the standard Python logger, and it follows a similar set of functionalities:

.. literalinclude:: code/logger.py
   :lines: 10-29

By default, the console only shows messages with the ``info`` level or higher, while the ``.log`` file
(if console logging is active) stores everything down to the ``debug`` level. You can change these
thresholds through the ``console_log_level`` and ``file_log_level`` arguments of the Logger constructor,
using the standard Python ``logging`` levels:

.. code-block:: python

    import logging
    logger = Logger('tutorial', results_dir='/tmp/logs', log_console=True,
                    console_log_level=logging.DEBUG, file_log_level=logging.DEBUG)

We can also log to terminal the exceptions. Using this method, instead of a raw print, you can manage
correctly the exception output without breaking any ``tqdm`` progress bar (see below), and the exception
text will be saved in the console log files (if console logging is active).

.. literalinclude:: code/logger.py
   :lines: 31-38

Logging a Reinforcement Learning experiment
-------------------------------------------

Our Logger includes some functionalities to log RL experiment data easily.
To demonstrate this, we will set up a simple RL experiment, using Q-Learning in the simple chain enviornment.

.. literalinclude:: code/logger.py
   :lines: 41-59

We skip the details of this RL experiment, as they are not relevant to the current tutorial.
You can have a deeper look at RL experiments with MushroomRL in other tutorials.

It is important to notice that we use ``tqdm`` progress bar, as our logger is integrated with
this package, and can print log messages while the progress bar is showing progress, without
disrupting the progress bar and the terminal.

We first print the learning performances before the learning, using the ``epoch_info`` method:

.. literalinclude:: code/logger.py
   :lines: 61-69

Notice that this method can print any possible label passed as a function parameter, so it's not
restricted to ``J``, ``R``, or other predefined metrics.

We now consider the learning loop:

.. literalinclude:: code/logger.py
   :lines: 70-87

Here we make use of both the ``epoch_info`` method to log the data in the console output and the methods
``log_numpy`` and ``log_best_agent`` to log the learning progress.

The ``log_numpy`` method can take an arbitrary value (primitive or a NumPy array) and log into a single NumPy array (or matrix). Again a set of arbitrary keywords can be used to save data into different filenames.
If the ``seed`` parameter of the constructor of the Logger class is specified, the filename will include
a postfix with the seed. This is useful when multiple runs of the same experiment are executed.

The ``log_best_agent`` saves the current agent, into the 'agent-best.msh' file. However, the current agent will
be stored on disk only if it improves w.r.t. the previously logged one.

We conclude the learning experiment by logging the final agent and the last dataset:


.. literalinclude:: code/logger.py
   :lines: 89-95


Advanced Logger topics
----------------------

The logger can be also used to continue the learning from a previously existing run, without overwriting the
stored results values. This can be done by specifying the ``append`` flag in the logger's constructor.

.. literalinclude:: code/logger.py
   :lines: 97-

Finally, another functionality of the logger is to activate some specific output from some algorithms.
This can be done by calling the ``set_logger`` method on the ``Core`` (or ``VectorCore``) object, which
forwards the logger to the agent and automatically configures the video recording fps from the
environment:

.. code-block:: python

    core.set_logger(logger)

Alternatively, the logger can be passed directly as a constructor argument:

.. code-block:: python

    core = Core(agent, mdp, logger=logger)

Algorithms use the logger to describe some learning metrics after every fit, both as console output and,
if enabled, as Weights & Biases logging, described next.


Logging to Weights & Biases
---------------------------

The Logger can optionally log to `Weights & Biases <https://wandb.ai>`_ (wandb), in addition to the
console and the numpy disk logging described above.

wandb logging is an optional functionality: it is enabled only if the ``wandb`` package is installed
and a set of init arguments is provided to the Logger. If ``wandb`` is not installed, or no init
arguments are provided, every wandb logging call is a safe no-op. You can install the optional
dependency with:

.. code-block:: bash

    pip install mushroom_rl[wandb]

To enable wandb logging, we build a dictionary of arguments for ``wandb.init`` and pass it to the
Logger through the ``wandb_kwargs`` argument. The helper static method ``default_wandb_kwargs``
returns an editable dictionary with sensible defaults; the ``config`` argument should contain the
experiment hyperparameters:

.. literalinclude:: code/wandb_logging.py
    :lines: 1-12

The ``log_training`` method logs the training metrics: they are grouped under the ``training/`` prefix in
wandb (using the number of fits as x-axis), printed on the console with the ``debug`` level (so they are
not shown by default), and stored on disk as numpy arrays inside a ``training`` subfolder only if the
Logger was constructed with ``force_numpy=True``. A ``'/'`` in a name groups the metric in wandb and is
replaced by ``'_'`` in the numpy file name:

.. literalinclude:: code/wandb_logging.py
    :lines: 14-15

The number of fits counter, used as x-axis for the training metrics, is advanced explicitly by calling
``advance_step`` once per fit, so that all the values logged during a fit share the same x-axis value:

.. literalinclude:: code/wandb_logging.py
    :lines: 17-19

The ``log_evaluation`` method logs the evaluation metrics: they are grouped under the ``eval/`` prefix in
wandb (using the epoch as x-axis), printed on the console through ``epoch_info``, and stored on disk as
numpy arrays in the logging directory:

.. literalinclude:: code/wandb_logging.py
    :lines: 21-22

When the Logger is created, it automatically sets the wandb ``group`` to the experiment name
(``log_name``) unless a ``group`` is already specified in ``wandb_kwargs``. This means that all runs
from the same experiment (e.g. different seeds) are grouped together in the wandb dashboard. You can
override this by passing an explicit ``group`` in ``wandb_kwargs``:

.. code-block:: python

    wandb_kwargs = Logger.default_wandb_kwargs('my_project', group='custom_group')

When a ``seed`` is passed to the Logger, it is automatically added to the wandb ``config`` dictionary
and, if ``name`` is not already set, the run name is set to ``log_name_seed`` (e.g. ``SAC_42``).
This makes it easy to distinguish individual seed runs within the same group:

The wandb run is finished automatically when the process exits, so there is usually no need to close it
explicitly; the ``finish`` method is available to close it earlier if needed.

While the numpy evaluation logging is typically driven by the experiment script, the training logging is
meant to be driven from inside the algorithms, which log their internal metrics (e.g. losses, entropy, KL
divergence) during the fit, once the logger is attached to the agent through the ``Core`` (via
``set_logger`` or the constructor) as shown above. A complete runnable example with metric logging is
available in ``examples/wandb_logging.py``.


Video Recording
---------------

The Logger includes a ``VideoLogger`` mixin that handles video recording. Videos are saved in a
``videos/`` subfolder of the logging directory. The recorder is created lazily on the first frame,
so no resources are allocated until recording actually starts.

To record during evaluation or learning, pass ``record=True`` (and ``render=True``) to the ``Core``
methods. The ``Core`` delegates recording to the agent's logger:

.. code-block:: python

    logger = Logger('my_experiment', results_dir='./logs')
    core = Core(agent, mdp, logger=logger)

    # Record a video during evaluation
    core.evaluate(n_episodes=1, render=True, record=True)

The fps is automatically set from the environment when the logger is attached to the ``Core`` via
``set_logger`` or the constructor. It can also be set explicitly in the Logger constructor:

.. code-block:: python

    logger = Logger('my_experiment', results_dir='./logs', fps=30)

By default, the ``VideoRecorder`` class from ``mushroom_rl.utils.record`` is used, which writes
``.mp4`` files using OpenCV with the VP9 codec. The codec can be changed through the
``recorder_kwargs`` argument (e.g. ``recorder_kwargs=dict(codec='avc1')`` for H.264).
A custom recorder class can be provided through the ``recorder_class`` argument.
The class must implement ``__call__(frame)`` and ``stop()`` methods:

.. code-block:: python

    logger = Logger('my_experiment', results_dir='./logs',
                    recorder_class=MyCustomRecorder)

The underlying recorder instance is accessible through ``logger.video_recorder`` after the first frame
has been recorded, and the list of recorded (and, with ``append=True``, previously stored) video files
is available through ``logger.recorded_videos``.

If wandb logging is active, the last recorded video can be uploaded to wandb through the ``log_video``
method. The recording is stopped by the ``Core``, so ``log_video`` only handles the upload, using the
epoch as x-axis. The video is uploaded under the ``video/`` group with a fixed key (``evaluation`` by
default), so that wandb shows a slider to browse videos across epochs:

.. code-block:: python

    core.evaluate(n_episodes=1, render=True, record=True)
    logger.log_video(epoch)

A specific video file can also be uploaded instead of the last recorded one by passing its path through
the ``video`` argument:

.. code-block:: python

    logger.log_video(epoch, video='./logs/my_experiment/videos/recording.mp4')


