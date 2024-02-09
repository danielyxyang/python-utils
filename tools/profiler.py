__all__ = ["Profiler"]

import contextlib
import time

import numpy as np


class Profiler():
    """Class for profiling execution time of code."""

    def __init__(self):
        # simple profiling
        self._start = None
        self._end = None
        # named profiling
        self._records = {}
        self.__names_context = [] # stack storing names of disjoint records for nested profiling contextmanagers

    def start(self, name=None):
        """Start timer of profiler."""
        if name is None:
            # simple profiling
            self._start = time.time()
            self._end = None
        else:
            # named profiling
            if name not in self._records:
                self._records[name] = {
                    "start": None,
                    "end": None,
                    "time": 0.0,
                    "info": None
                }
            self._records[name]["start"] = time.time()
            self._records[name]["end"] = None

    def stop(self, name=None):
        """Stop timer of profiler and save time under given name cumulatively."""
        if name is None:
            # simple profiling
            if self._start is None:
                print("WARNING: profiler has not been started yet")
                return
            self._end = time.time()
        else:
            # named profiling
            if name not in self._records:
                print("WARNING: profiler has not been started for \"{}\" yet".format(name))
                return
            self._records[name]["end"] = time.time()
            self._records[name]["time"] += self.time(name)

    def time(self, name=None):
        """Return last stopped time."""
        if name is None:
            # simple profiling
            if self._start is None or self._end is None:
                print("WARNING: profiler has not recorded a time yet")
                return
            return self._end - self._start
        else:
            # named profiling
            if name not in self._records or self._records[name]["start"] is None or self._records[name]["end"] is None:
                print("WARNING: profiler has not recorded a time for \"{}\" yet".format(name))
                return
            return self._records[name]["end"] - self._records[name]["start"]

    def set_info(self, name, info):
        """Set profiling information for the stopped time under given name."""
        self._records[name]["info"] = info

    def merge(self, profiler):
        """Merge records with another profiler."""
        self._records.update(profiler._records)

    def reset(self):
        """Reset profiler and delete all records."""
        self._start = None
        self._end = None
        self._records.clear()

    def print(self, names=None):
        """Print saved times and profiling information for the given list of names."""
        if names is None:
            names = self._records.keys()

        print("Profiling")
        max_length = np.max([len(name) for name in names])
        total = 0
        for name in names:
            time = self._records[name]["time"] if name in self._records else None
            info = self._records[name]["info"] if name in self._records else None
            total += time if time is not None else 0
            print("  {:{}} {}{}".format(
                "{}:".format(name), max_length + 1,
                "{:4.2f}s".format(time) if time is not None else "{:4} ".format("-"),
                " ({})".format(info) if info is not None else "",
            ))
        print("  {:{}} {}".format("Total:", max_length + 1, "{:4.2f}s".format(total)))

    @contextlib.contextmanager
    def cm(self, name=None, disjoint=True):
        """Create context manager for profiling.

        Args:
            name (str): Name of the profiling session.
            disjoint (bool): Flag whether recorded time should be disjoint from
                other times recorded with context manager. Defaults to True."""
        if name is None:
            # simple profiling
            self.start()
            try:
                yield self
            finally:
                self.stop()
        else:
            # named profiling
            # stop previous session
            if len(self.__names_context) > 0:
                name_prev = self.__names_context[-1]
                self.stop(name=name_prev)
            # start this session
            if disjoint:
                self.__names_context.append(name)
            self.start(name=name)
            try:
                yield self
            finally:
                # stop this session
                self.stop(name=name)
                if disjoint:
                    self.__names_context.pop()
                # start previous session
                if len(self.__names_context) > 0:
                    name_prev = self.__names_context[-1]
                    self.start(name=name_prev)
