from pathlib import Path
import json
import logging
import pickle
import types
from collections import UserDict
from collections.abc import Mapping, MutableMapping

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
            logger.warning(f"LoopChecker in \"{self.location}\" reached threshold {self.threshold}.")


class LazyDict(UserDict):
    """Class for lazily evaluating function items of dict on first access."""
    def __getitem__(self, key):
        if isinstance(self.data[key], types.FunctionType):
            self.data[key] = self.data[key]()
        return self.data[key]


def to_dict(obj):
    """Convert an object into a nested dictionary.

    The code is taken from [1].

    References:
        [1] https://stackoverflow.com/a/22679824
    """
    if isinstance(obj, str):
        return obj
    elif hasattr(obj, "__dict__"):
        return to_dict(vars(obj))
    elif isinstance(obj, dict):
        return {key: to_dict(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [to_dict(val) for val in obj]
    else:
        return obj


def flatten_dict(d, parent_key=None, sep='/'):
    """Flatten a dict with the given separator and under the given parent_key.

    The code is taken from [1].

    References:
        [1] https://www.freecodecamp.org/news/how-to-flatten-a-dictionary-in-python-in-4-different-ways/
    """
    def _flatten_dict_gen(d, parent_key, sep):
        for key, value in d.items():
            new_key = parent_key + sep + key if parent_key is not None else key
            if isinstance(value, MutableMapping):
                yield from flatten_dict(value, new_key, sep=sep).items()
            else:
                yield new_key, value
    return dict(_flatten_dict_gen(d, parent_key, sep))


def update_dict(d, u):
    """Update a nested dict with the given nested dict.

    The code is taken from [1].

    References:
        [1] https://stackoverflow.com/a/3233356
    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


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


def use_cache(f, path, refresh=False, verbose=True):
    path = Path(path)

    if not path.is_file() or refresh:
        # evaluate function
        output = f()
        # save to disk
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".npy":
            np.save(path, output)
        elif path.suffix == ".npz":
            np.savez_compressed(path, **output)
        elif path.suffix == ".pkl":
            with open(path, "wb") as f:
                pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f"Unknown file format \"{path.suffix}\".")
        if verbose:
            logger.info(f"Saved \"{path}\" to cache.")
    else:
        # load from disk
        if path.suffix in [".npy", ".npz"]:
            output = np.load(path)
        elif path.suffix == ".pkl":
            with open(path, "rb") as f:
                output = pickle.load(f)
        else:
            raise ValueError(f"Unknown file format \"{path.suffix}\".")
        if verbose:
            logger.info(f"Loaded \"{path}\" from cache.")
    return output
