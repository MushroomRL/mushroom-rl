from datetime import datetime
from pathlib import Path

from .console_logger import ConsoleLogger
from .data_logger import DataLogger


class Logger(DataLogger, ConsoleLogger):
    """
    This class implements the logging functionality. It can be used to create
    automatically a log directory, save numpy data array and the current agent.

    """
    def __init__(self, log_name='', results_dir='./logs', log_console=False,
                 use_timestamp=False, append=False, seed=None, **kwargs):
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

        DataLogger.__init__(self, results_dir, suffix=suffix, append=append)
        ConsoleLogger.__init__(self, log_name, results_dir if log_console else None,
                               suffix=suffix, **kwargs)
