import contextlib
import logging
import sys


def setup_logging(level=logging.INFO, format=None, dateformat=None, other_loggers=[]):
    if format is None:
        format="[%(asctime)s] %(levelname)-8s %(name)s:%(lineno)-5d %(message)s"
    if dateformat is None:
        dateformat="%Y-%m-%d %H:%M:%S"

    # configure logging system
    logging.basicConfig(
        stream=sys.stdout, # log to stdout instead of stderr to sync with print()
        level=level,
        format=format,
        datefmt=dateformat,
    )
    logging.captureWarnings(True)

    # configure other loggers
    for logger in other_loggers:
        if isinstance(logger, str):
            logger = logging.getLogger(logger)
        for handler in logger.handlers:
            handler.setFormatter(logging.Formatter(format, dateformat))
            handler.setLevel(level)

# reference: https://gist.github.com/simon-weber/7853144
@contextlib.contextmanager
def logging_disabled(level=logging.CRITICAL):
    prev_root_level = logging.root.manager.disable
    logging.disable(level)
    try:
        yield
    finally:
        logging.disable(prev_root_level)
