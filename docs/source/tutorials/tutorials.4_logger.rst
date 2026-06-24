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
This can be done by calling the agent's ``set_logger`` method:

.. code-block:: python

    agent.set_logger(logger)

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

The unified ``log`` method logs a set of named values to every active backend. By default, the values
are sent to wandb and printed on the console with the ``debug`` level (so they are not shown by
default), but they are not stored on disk. To also save the values on disk as numpy arrays, construct
the Logger with ``force_numpy=True``:

.. literalinclude:: code/wandb_logging.py
    :lines: 14-15

wandb associates each logged value with a monotonically increasing step. The Logger keeps an internal
step counter that is shared by all the values logged through ``log``. The counter is advanced explicitly
by calling ``advance_step``, typically once a logical step (e.g. an epoch, or an algorithm update) is
complete:

.. literalinclude:: code/wandb_logging.py
    :lines: 17-19

While the numpy logging is typically driven by the experiment script, wandb logging is meant to be
driven from inside the algorithms, which log their internal metrics (e.g. losses, entropy, KL
divergence) during the fit, once the logger is passed to the agent via ``set_logger`` as shown above.


