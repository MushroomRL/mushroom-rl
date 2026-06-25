from datetime import datetime
from pathlib import Path

from mushroom_rl.core.logger.console_logger import ConsoleLogger
from mushroom_rl.core.logger.data_logger import DataLogger
from mushroom_rl.core.logger.video_logger import VideoLogger
from mushroom_rl.core.logger.wandb_logger import WandbLogger


class Logger(DataLogger, ConsoleLogger, VideoLogger, WandbLogger):
    """
    This class implements the logging functionality. It can be used to create
    automatically a log directory, save numpy data array and the current agent.
    It optionally logs to Weights & Biases (wandb), if the ``wandb`` package is
    installed and a set of init arguments is provided.

    """
    def __init__(self, log_name='', results_dir='./logs', log_console=False,
                 use_timestamp=False, append=False, seed=None, wandb_kwargs=None,
                 force_numpy=False, recorder_class=None, fps=None, recorder_kwargs=None,
                 **kwargs):
        """
        Constructor.

        Args:
            log_name (string, ''): name of the current experiment directory if not
                specified, the current timestamp is used.
            results_dir (string, './logs'): name of the base logging directory.
                If set to None, no directory is created;
            log_console (bool, False): whether to log or not the console output;
            use_timestamp (bool, False): If true, adds the current timestamp to
                the folder name;
            append (bool, False): If true, the logger will append the new
                data logged to the one already existing in the directory;
            seed (int, None): seed for the current run. It can be optionally
                specified to add a seed suffix for each data file logged;
            wandb_kwargs (dict, None): dictionary of arguments forwarded to
                ``wandb.init`` to enable wandb logging. If None, or if the
                ``wandb`` package is not installed, wandb logging is disabled.
                Use ``Logger.default_wandb_kwargs`` to build a default dictionary;
            force_numpy (bool, False): if True, the values logged through the
                ``log`` method are also stored on disk as numpy arrays (only if a
                results directory is set);
            recorder_class (class, None): the class used to record video. By default,
                the ``VideoRecorder`` class is used. The class must implement the
                ``__call__`` and ``stop`` methods;
            fps (int, None): frames per second for video recording. If None, the
                value is set automatically by ``Core.set_logger`` from the environment;
            recorder_kwargs (dict, None): additional keyword arguments forwarded to
                the recorder class constructor;
            **kwargs: other parameters for ConsoleLogger class.

        """

        if log_console:
            assert results_dir is not None

        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        if not log_name:
            log_name = timestamp
        elif use_timestamp:
            log_name += '_' + timestamp

        if results_dir:
            results_dir = Path(results_dir) / log_name
            results_dir.mkdir(parents=True, exist_ok=True)

        suffix = '' if seed is None else '-' + str(seed)

        self._force_numpy = force_numpy and results_dir is not None

        video_path = results_dir / 'videos' if results_dir else None

        DataLogger.__init__(self, results_dir, suffix=suffix, append=append)
        ConsoleLogger.__init__(self, log_name, results_dir if log_console else None,
                               suffix=suffix, **kwargs)
        VideoLogger.__init__(self, recorder_class=recorder_class, fps=fps,
                             video_path=video_path, **(recorder_kwargs or {}))
        WandbLogger.__init__(self, wandb_kwargs)

    def log(self, **kwargs):
        """
        Log a set of named scalars to every active logging backend. The values
        are always logged to wandb (if active) and to the console with the
        ``debug`` level (so they are not shown by default). They are logged to
        disk as numpy arrays only if the logger was constructed with
        ``force_numpy=True``.

        Args:
            **kwargs: set of named values to be logged. The argument name is used
                as label across all the backends.

        """
        self.log_wandb(**kwargs)

        if self._force_numpy:
            self.log_numpy(**kwargs)

        self.debug(' '.join(f'{name}: {data}' for name, data in kwargs.items()))
