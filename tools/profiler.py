import contextlib
import logging
import time

import numpy as np

try:
    import torch # type: ignore[import]
    _IMPORTED_TORCH = True
except ImportError:
    _IMPORTED_TORCH = False

logger = logging.getLogger(__name__)

class Timer:
    """Class for measuring execution time of one code section repeatedly."""

    def __init__(self):
        self.times = []

    # Basic methods for measuring time

    def is_started(self):
        """Check if timer has been started."""
        return hasattr(self, "_time_start")

    def is_running(self):
        """Check if timer is running."""
        return self.is_started() and not hasattr(self, "_time_paused")

    def is_paused(self):
        """Check if timer is paused."""
        return self.is_started() and hasattr(self, "_time_paused")

    @property
    def current_time(self):
        """Current synchronized time."""
        if _IMPORTED_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def start(self):
        """Start timer."""
        if self.is_started():
            raise RuntimeError("Timer has already been started.")

        self._time_start = self.current_time

    def stop(self):
        """Stop timer and save measured time."""
        if not self.is_started():
            raise RuntimeError("Timer has not been started yet.")

        self.times.append(self.current_time - self._time_start)
        del self._time_start

    def pause(self):
        """Pause timer and save current time."""
        if self.is_paused():
            raise RuntimeError("Timer is already paused.")

        self._time_paused = self.current_time

    def resume(self):
        """Resume timer from the last paused time."""
        if self.is_running():
            raise RuntimeError("Timer is already running.")

        self._time_start += self.current_time - self._time_paused
        del self._time_paused

    def reset(self):
        """Reset timer and clear all measured times."""
        self.times.clear()

    # Methods for context manager

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    # Properties for accessing time statistics

    @property
    def num_times(self):
        """Number of measured times."""
        return len(self.times)

    @property
    def mean(self):
        """Mean of measured times."""
        return np.mean(self.times)

    @property
    def std(self):
        """Standard deviation of measured times."""
        return np.std(self.times)

    @property
    def total(self):
        """Sum of measured times."""
        return np.sum(self.times)

    def __str__(self):
        if self.num_times == 0:
            return "-"
        if self.num_times == 1:
            return f"{self.mean:.4f}s"
        else:
            return f"{self.mean:.4f}s ± {self.std:.4f}s ({self.num_times} times), total {self.total:.4f}s"


class TimeProfiler:
    """Class for profiling execution times of multiple code sections."""

    def __init__(self):
        self.timers = {}
        self.__names_context = []  # stack storing names of disjoint nested timers

    # Basic methods for profiling

    def start(self, name):
        """Start timer under given name."""
        if name not in self.timers:
            self.timers[name] = Timer()
        self.timers[name].start()

    def stop(self, name):
        """Stop timer under given name."""
        if name not in self.timers:
            raise RuntimeError(f"Timer \"{name}\" has not been started yet.")
        self.timers[name].stop()

    def pause(self, name):
        """Pause timer under given name."""
        if name not in self.timers:
            raise RuntimeError(f"Timer \"{name}\" has not been started yet.")
        self.timers[name].pause()

    def resume(self, name):
        """Resume timer under given name."""
        if name not in self.timers:
            raise RuntimeError(f"Timer \"{name}\" has not been started yet.")
        self.timers[name].resume()

    def reset(self):
        """Reset all timers."""
        self.timers.clear()

    # Methods for context manager

    @contextlib.contextmanager
    def __call__(self, name, disjoint=False):
        """Create context manager for profiling.

        Args:
            name (str): Name of the profiling session.
            disjoint (bool, optional): Flag whether this measured time should be
                disjoint from nested times measured with this context manager.
                Defaults to False."""
        # pause parent session
        if len(self.__names_context) > 0:
            name_prev = self.__names_context[-1]
            self.pause(name_prev)
        # start this session
        if disjoint:
            self.__names_context.append(name)
        self.start(name)
        try:
            yield self
        finally:
            # stop this session
            self.stop(name)
            if disjoint:
                self.__names_context.pop()
            # resume parent session
            if len(self.__names_context) > 0:
                name_prev = self.__names_context[-1]
                self.resume(name_prev)

    # Methods for accessing profiling information

    def summary(self, names="all"):
        """Print saved times and profiling information for the given list of names."""
        if names == "all":
            names = self.timers.keys()

        offset = np.max([len(name) for name in names]) + 2
        summary = "TimeProfiler"
        for name in names:
            summary += f"\n  {name + ':':{offset + 1}}{self.timers.get(name, '-')}"

        return summary

    def __str__(self):
        return self.summary()
