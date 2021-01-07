import logging
import tqdm


class TqdmHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.NOTSET)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


class ConsoleLogger(object):
    """
    This class implements the console logging functionality. It can be used to
    log text into the console and optionally save a log file.

    """
    def __init__(self, log_name, log_dir=None, suffix='',
                 log_file_name=None,
                 console_log_level=logging.DEBUG,
                 file_log_level=logging.DEBUG):
        """
        Constructor.

        Args:
            log_name (str, None): Name of the current logger.
            log_dir (Path, None): path of the logging directory. If None, no
                the console output is not logged into a file;
            suffix (int, None): optional string to add a suffix to the logger id
                and to the data file logged;
            log_file_name (str, None): optional specifier for log file name,
                id is used by default;
            console_log_level (int, logging.DEBUG): logging level for console;
            file_log_level (int, logging.DEBUG): logging level for file.

        """
        self._log_id = log_name + suffix

        formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)s] %(message)s',
                                      datefmt='%d/%m/%Y %H:%M:%S')

        self._logger = logging.getLogger(self._log_id)
        self._logger.setLevel(min(console_log_level, file_log_level))
        self._logger.propagate = False
        ch = TqdmHandler()
        ch.setLevel(console_log_level)
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)

        if log_dir is not None:
            log_file_name = self._log_id if log_file_name is None else log_file_name
            log_file_name += '.log'
            log_file_path = log_dir / log_file_name
            fh = logging.FileHandler(log_file_path)
            fh.setLevel(file_log_level)
            fh.setFormatter(formatter)
            self._logger.addHandler(fh)

    def debug(self, msg):
        """
        Log a message with DEBUG level

        """
        self._logger.debug(msg)

    def info(self, msg):
        """
        Log a message with INFO level

        """
        self._logger.info(msg)

    def warning(self, msg):
        """
        Log a message with WARNING level

        """
        self._logger.warning(msg)

    def error(self, msg):
        """
        Log a message with ERROR level

        """
        self._logger.error(msg)

    def critical(self, msg):
        """
        Log a message with CRITICAL level

        """
        self._logger.critical(msg)

    def exception(self, msg):
        """
        Log a message with ERROR level. To be called
        only from an exception handler

        """
        self._logger.exception(msg)

    def strong_line(self):
        """
        Log a line of #

        """
        self.info('###################################################################################################')

    def weak_line(self):
        """
        Log a line of -

        """
        self.info('---------------------------------------------------------------------------------------------------')

    def epoch_info(self, epoch, **kwargs):
        """
        Log the epoch info with the format: Epoch <epoch number> | <label 1>: <data 1> <label 2> <data 2> ...

        Args:
            epoch (int): epoch number;
            **kwargs: the labels and the data to be displayed.

        """
        msg = 'Epoch ' + str(epoch) + ' |'

        for name, data in kwargs.items():
            msg += ' ' + name + ': ' + str(data)

        self.info(msg)

    def __del__(self):
        self._logger.handlers.clear()
