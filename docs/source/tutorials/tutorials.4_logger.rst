How to use the Mushroom Logger
==============================

Here we explain in a bit of detail the usage of the MushroomRL Logger class.
This class can be used as a standardized console logger and can also log on disk
NumPy arrays or a mushroom agent, using the appropriate logging folder.


Constructing the Logger
-----------------------

To initialize the logger we can simply choose a log directory and an experiment name:

.. literalinclude:: code/logger.py
    :lines: 1-5

This will create the experiment folder named 'tutorial' inside the base folder '/tmp/logs'.
The logger creates all the necessary directories if they don't exist.
If results_dir is not specified, the log will create a './logs' base directory.
By setting ``log_console`` to true, the logger will store the console output in a '.log' text file inside the experiment folder, with the same name.
If the file already exists, the logger will append the new logged lines.

If you don't want the logger to create any directory e.g., to only use the log for the console
output, you can force the ``results_dir`` parameter to None:

.. literalinclude:: code/logger.py
   :lines: 7-8

Logging message on the console
------------------------------

The most basic functionality of the logger is to output text messages on the standard output.
Our logger uses the standard python logger, and it follows a similar set of functionality:

.. literalinclude:: code/logger.py
   :lines: 10-29

We can also log to the terminal the exceptions. Using this method, instead of a raw print, you can manage
correctly the exception output without breaking any ``tqdm`` progress bar (see below), and the exception
text will be saved in the console log files (if console logging is active).

.. literalinclude:: code/logger.py
   :lines: 31-38

Logging a Reinforcement Learning Experiment
-------------------------------------------

Our logger includes some functionalities to log RL experiment data easily.
To demonstrate this, we will set up a simple RL experiment, using Q-Learning in the simple chain enviornment.

.. literalinclude:: code/logger.py
   :lines: 41-59

We will skip the details of this RL experiment, as they are not relevant to the current tutorial.
You can have a deeper look at RL experiments with Mushroom in the previous tutorials.

It's important to notice that we will use ``tqdm`` progress bar, as our logger is integrated with
this package, and can print log messages while the progress bar is showing progress, without
disrupting the progress bar and the terminal.

We will first print the learning performances before the learning, using the epoch_info method:

.. literalinclude:: code/logger.py
   :lines: 61-69

Notice that this method can print any possible label passed as a function parameter, so it's not
restricted to J, R, or other predefined metrics.

We will now consider the learning loop:

.. literalinclude:: code/logger.py
   :lines: 71-87

Here we make use of both the ``epoch_info`` method to log the data in the console output and the methods
``log_numpy`` and ``log_best_agent`` to log the learning progress.

The ``log_numpy`` method can take an arbitrary value (primitive or a NumPy array) and log into a single NumPy array (or matrix). Again a set of arbitrary keywords can be used to save data into different filenames.
If the ``seed`` parameter of the constructor of the Logger class is specified, the filename will include
a postfix with the seed. This is useful when multiple runs of the same experiment are executed.

The ``log_best_agent`` save the current agent, into the 'agent-best.msh' file. However, the current agent will
be stored on disk only if it improves w.r.t. the previously logged one.

We conclude the learning experiment by logging the final agent:


.. literalinclude:: code/logger.py
   :lines: 89-92


Advanced Logger topics
----------------------

The logger can be also used to continue the learning from a previously existing run, without overwriting the
stored results values. This can be done by specifying the ``append`` flag in the logger's constructor.

.. literalinclude:: code/logger.py
   :lines: 89-92

Finally, another functionality of the logger is to activate some specific text output from some algorithms.
This can be done by calling the agent's ``set_logger`` method:

.. code-block:: python

    agent.set_logger(logger)


Currently, only the ``PPO`` and the ``TRPO`` algorithms provide additional output, by describing some
learning metrics after every fit.


