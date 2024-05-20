import json
import logging
import types
from collections import UserDict
from collections.abc import MutableMapping

import numpy as np

logger = logging.getLogger(__name__)

class LoopChecker():
    """Class for checking loops on non-termination."""
    def __init__(self, location=None, threshold=1e6):
        self.location = location
        self.threshold = int(threshold)
        self.counter = 0

    def __call__(self):
        self.counter = (self.counter + 1) % self.threshold
        if(self.counter == 0):
            logger.warning("LoopChecker in \"{}\" reached threshold {}".format(self.location, self.threshold))


class LazyDict(UserDict):
    """Class for lazily evaluating function items of dict on first access."""
    def __getitem__(self, key):
        if isinstance(self.data[key], types.FunctionType):
            self.data[key] = self.data[key]()
        return self.data[key]


def _flatten_dict_gen(d, parent_key, sep):
    for key, value in d.items():
        new_key = parent_key + sep + key if parent_key is not None else key
        if isinstance(value, MutableMapping):
            yield from flatten_dict(value, new_key, sep=sep).items()
        else:
            yield new_key, value

def flatten_dict(d, parent_key=None, sep='/'):
    """Flatten a dict with the given separator and under the given parent_key.

    The code is taken from [1].

    References:
        [1] https://www.freecodecamp.org/news/how-to-flatten-a-dictionary-in-python-in-4-different-ways/
    """
    return dict(_flatten_dict_gen(d, parent_key, sep))


def build_json_encoder(encoders=[]):
    """Encode common non-serializable types into serializable ones."""
    def encoder(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        for instance, encoder in encoders:
            if isinstance(obj, instance):
                return encoder(obj)
        return json.JSONEncoder().default(obj)
    return encoder
