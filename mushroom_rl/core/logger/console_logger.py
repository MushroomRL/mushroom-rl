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
    def __init__(self, log_dir=None, suffix='',
                 console_log_level=logging.DEBUG,
                 file_log_level=logging.DEBUG):
        """
        Constructor.

        Args:
            log_dir (Path): path of the logging directory. If None, no
                the console output is not logged into a file;
            suffix (int, None): optional string to add a suffix to the logger id
                and to the data file logged;
            console_log_level (int, logging.DEBUG): logging level for console;
            file_log_level (int, logging.DEBUG): logging level for file.

        """
        self._log_id = 'log' + suffix

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
            log_file_name = self._log_id + '.log'
            log_file_path = log_dir / log_file_name
            fh = logging.FileHandler(log_file_path)
            fh.setLevel(file_log_level)
            fh.setFormatter(formatter)
            self._logger.addHandler(fh)

    def debug(self, msg):
        self._logger.debug(msg)

    def info(self, msg):
        self._logger.info(msg)

    def warn(self, msg):
        self._logger.warn(msg)

    def error(self, msg):
        self._logger.error(msg)

    def strong_line(self):
        self.info('###################################################################################################')

    def weak_line(self):
        self.info('---------------------------------------------------------------------------------------------------')

    def epoch_info(self, epoch, **kwargs):
        msg = 'Epoch ' + str(epoch) + ' |'

        for name, data in kwargs.items():
            msg += ' ' + name + ': ' + str(data)

        self.info(msg)
