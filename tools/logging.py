__all__ = [
    "setup_logging",
    "logging_disabled",
]

import contextlib
import logging
import sys


def setup_logging():
    logging.basicConfig(
        stream=sys.stdout, # log to stdout instead of stderr to sync with print()
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s:%(lineno)-5d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.captureWarnings(True)

# reference: https://gist.github.com/simon-weber/7853144
@contextlib.contextmanager
def logging_disabled(level=logging.CRITICAL):
    prev_root_level = logging.root.manager.disable
    logging.disable(level)
    try:
        yield
    finally:
        logging.disable(prev_root_level)
