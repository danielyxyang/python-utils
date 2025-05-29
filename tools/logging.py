import contextlib
import glob
import logging
import os
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

def parse_log(path, patterns, transform={}, repeat=False):
    r"""Parse log files based on the given list of patterns.

    Examples:
        Parsing basic log file
        ```
        duration = lambda s: datetime.strptime(s, "%H:%M:%S") - datetime(1900, 1, 1)
        timestamp = lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        log = parse_logs("path/to/file.log", {
            "string":    (r"A string:    (.+)", None),
            "int":       (r"An integer:  (\d+)", int),
            "float":     (r"A float:     ([+-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?)", float),
            "duration":  (r"A duration:  ([0-9\:]+)", duration),
            "timestamp": (r"A timestamp: ([0-9\:\- ]+)", timestamp),
        })
        ```

    Args:
        path (str): The path to the log file.
        patterns (dict): The dictionary mapping pattern name to a tuple with the
            regex string and the function used to transform the parsed value.
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
    patterns = [
        (
            pattern_name,
            (re.compile(pattern, re.MULTILINE), transform if transform is not None else (lambda s: s)),
        )
        for pattern_name, (pattern, transform) in patterns.items()
    ]

    # find matches
    matches = [{pattern_name: None for pattern_name, _ in patterns}]
    next_search_pos = 0
    next_pattern_index = 0
    while next_search_pos < len(logs):
        # search for pattern
        pattern_name, (pattern, transform) = patterns[next_pattern_index]
        match = pattern.search(logs, pos=next_search_pos)
        if match is None:
            break
        # add match
        if len(match.groups()) == 1:
            matches[-1][pattern_name] = transform(match.group(1))
        else:
            matches[-1][pattern_name] = transform(match.groups())
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

def parse_logs(path, patterns, repeat=False, path_transform=None):
    r"""Parse multiple log files based on the given list of patterns.

    Examples:
        Parsing and aggregating multiple log files
        ```
        logs = parse_logs(
            "path/to/*.log",
            patterns={
                "string":    (r"A string:    (.+)", None),
                "int":       (r"An integer:  (\d+)", int),
                "float":     (r"A float:     ([+-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?)", float),
                "duration":  (r"A duration:  ([0-9\:]+)", duration),
                "timestamp": (r"A timestamp: ([0-9\:\- ]+)", timestamp),
            },
            repeat=True,
            path_transform=os.path.basename,
        )
        logs = pd.DataFrame(logs)
        logs = logs.groupby("path").aggregate({
            "string":   ("string", "first"),
            "int":      ("int", "sum"),
            "float":    ("float", "mean"),
            "duration": ("duration", "max"),
        })
        display(logs)
        ```

    Args:
        path (str): See parse_log.
        patterns (dict): See parse_log.
        repeat (bool, optional): See parse_log. Defaults to False.
        path_transform (function, optional): The function used to transform the
            path of the log file. Defaults to None.

    Returns:
        list of dicts: See parse_log.
    """
    if path_transform is None:
        path_transform = lambda path: path

    logs = []
    for path in sorted(glob.glob(str(path))):
        log = parse_log(path, patterns, repeat=repeat)
        if repeat:
            for l in log:
                l["path"] = path_transform(path)
            logs += log
        else:
            log["path"] = path_transform(path)
            logs.append(log)
    return logs
