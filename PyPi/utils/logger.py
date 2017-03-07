import logging


class Logger(object):
    def __init__(self, level):
        self._levels = [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL
        ]

        self.logger = logging.getLogger('logger')
        self.logger.setLevel(self._levels[level])
