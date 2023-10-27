__all__ = ["LoopChecker", "LazyDict", "CustomFormatter", "build_json_encoder"]

import json
import types
from string import Formatter
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


class CustomFormatter(Formatter):
    def __init__(self, format_funcs={}):
        super().__init__()
        self.format_funcs = format_funcs
    
    def get_field(self, field_name, args, kwargs):
        if (field_name.startswith("\'") and field_name.endswith("\'")) \
        or (field_name.startswith("\"") and field_name.endswith("\"")):
            # return inline string
            return field_name[1:-1], None
        else:
            # return field value
            return super().get_field(field_name, args, kwargs)
    
    def format_field(self, value, format_specs):
        # iterate through pipeline of formatting specifications
        format_specs = format_specs.split(":")
        for format_spec in format_specs:
            # check if formatting function should be applied elementwise or not
            if format_spec.startswith("@"):
                format_elementwise, format_spec = True, format_spec[1:]
            else:
                format_elementwise, format_spec = False, format_spec
            # define formatting function
            if format_spec in self.format_funcs:
                format_func = self.format_funcs[format_spec]
            else:
                format_func = lambda v: super(CustomFormatter, self).format_field(v, format_spec)
            # apply formatting function
            value = [format_func(v) for v in value] if format_elementwise else format_func(value)
        return value


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
