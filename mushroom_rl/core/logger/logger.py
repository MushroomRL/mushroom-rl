from datetime import datetime
from pathlib import Path

from .console_logger import ConsoleLogger
from .data_logger import DataLogger


class Logger(DataLogger, ConsoleLogger):
    """
    This class implements the logging functionality. It can be used to create
    automatically a log directory, save numpy data array and the current agent.

    """
    def __init__(self, results_dir=None, log_console=False,
                 use_timestamp=False, append=False, seed=None, **kwargs):
        """
        Constructor.

        Args:
            results_dir (string, None): name of the logging directory. If
                not specified, a time-stamped directory is created inside
                a 'log' folder;
            log_console (bool, False): whether to log or not the console output;
            use_timestamp (bool, False): If true, adds the current timestamp to
                the folder name;
            append (bool, False): If true, the logger will append the new
                data logged to the one already existing in the directory;
            seed (int, None): seed for the current run. It can be optionally
                specified to add a seed suffix for each data file logged;
            **kwargs: other parameters for ConsoleLogger class.

        """

        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        if results_dir is None:
            results_dir = Path('.', 'logs') / timestamp
        else:
            if use_timestamp:
                results_dir += '_' + timestamp
            results_dir = Path(results_dir)

        print('Logging in folder: ' + results_dir.name)
        results_dir.mkdir(parents=True, exist_ok=True)

        suffix = '' if seed is None else '-' + str(seed)
        
        DataLogger.__init__(self, results_dir, suffix=suffix, append=append)
        ConsoleLogger.__init__(self, results_dir if log_console else None,
                               suffix=suffix, **kwargs)
