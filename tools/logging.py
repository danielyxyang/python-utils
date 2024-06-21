import contextlib
import logging
import re
import sys

logger = logging.getLogger(__name__)

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

def parse_logs(path, patterns, repeat=False):
    # load log file
    with open(path) as f:
        logs = f.read()

    # load patterns
    patterns = [(pattern_name, re.compile(pattern, re.MULTILINE)) for pattern_name, pattern in patterns.items()]

    # find matches
    matches = [{pattern_name: None for pattern_name, _ in patterns}]
    next_search_pos = 0
    next_pattern_index = 0
    while next_search_pos < len(logs):
        # search for pattern
        pattern_name, pattern = patterns[next_pattern_index]
        match = pattern.search(logs, pos=next_search_pos)
        if match is None:
            break
        # add match
        matches[-1][pattern_name] = match.group(1) if len(match.groups()) == 1 else match.groups()
        # move to next search position
        next_search_pos = match.end()
        # move to next pattern
        next_pattern_index += 1
        if next_pattern_index == len(patterns):
            if repeat:
                # cycle through patterns
                matches.append({pattern_name: None for pattern_name, _ in patterns})
                next_pattern_index = 0
            else:
                break

    if repeat and next_pattern_index == 0 and len(matches) > 1:
        matches.pop(-1)
        next_pattern_index = len(patterns)

    if next_pattern_index < len(patterns):
        logger.warning(f"The following match in \"{path}\" is incomplete: {matches[-1]}")

    return matches if repeat else matches[0]
