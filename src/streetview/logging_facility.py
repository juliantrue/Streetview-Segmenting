import sys, os
import logging
import datetime

module_name = 'Streetview_Module'
debug_mode = True

class LoggingWrapper(object):

    def __init__(self, log_folder_path=None):
        self.debug_mode = debug_mode

        # Create logger with module name
        logger = logging.getLogger(module_name)
        logger.setLevel(logging.DEBUG)

        # create file handler which logs even debug messages
        now = datetime.datetime.now()
        log_file = '{}{}{}{}{}{}.log'.format(now.year, now.month, now.day,
                                                  now.hour, now.minute,
                                                  now.second)
        # If no folder provided, output to stderr
        if log_folder_path == None:
            fh = logging.StreamHandler(sys.stderr)
        else:
            log_file = os.path.join(log_folder_path, log_file)
            fh = logging.FileHandler(log_file)

        fh.setLevel(logging.DEBUG)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
