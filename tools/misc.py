__all__ = ["LoopChecker", "LazyDict", "build_json_encoder"]

import json
import types
from collections import UserDict

import numpy as np


class LoopChecker():
    """Class for checking loops on non-termination."""
    def __init__(self, location=None, threshold=1e6):
        self.location = location
        self.threshold = int(threshold)
        self.counter = 0
    
    def __call__(self):
        self.counter = (self.counter + 1) % self.threshold
        if(self.counter == 0):
            print("WARNING: LoopChecker in \"{}\" reached threshold {}".format(self.location, self.threshold))


class LazyDict(UserDict):
    """Class for lazily evaluating function items of dict on first access."""
    def __getitem__(self, key):
        if isinstance(self.data[key], types.FunctionType):
            self.data[key] = self.data[key]()
        return self.data[key]


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
