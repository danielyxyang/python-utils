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
    """Parse log files based on the given list of patterns.

    Examples:
        Parsing basic log file
        ```
        log = parse_logs("path/to/file.log", {
            "int":       r"An integer:  (\d+)",
            "float":     r"A float:     (\d+(?:\.\d*)?)",
            "duration":  r"A duration:  ([0-9\:]+)",
            "timestamp": r"A timestamp: ([0-9\:\- ]+)",
        })
        log["duration"] = datetime.strptime(log["duration"], "%H:%M:%S") - datetime(1900, 1, 1)
        log["timestamp"] = datetime.strptime(log["timestamp"], "%Y-%m-%d %H:%M:%S")
        ```

        Parsing multiple log files
        ```
        from utils_ext.widgets import display_table

        logs = [parse_logs(path, ...) for path in sorted(glob.glob("path/to/*.log"))]
        logs = pd.DataFrame(logs)
        display_table(logs_grouped)
        ```

    Args:
        path (str): The path to the log file.
        patterns (dict): The dictionary mapping pattern name to a regex string.
        repeat (bool, optional): The flag whether the log should be repeatedly
            parsed for the given patterns or not. Defaults to False.

    Returns:
        dict or list of dicts: The dictionary mapping pattern name to the parsed
            group or list of groups. If repeat is True, a list of such
            dictionaries will be returned.
    """
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
