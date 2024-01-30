__all__ = [
    "LoopChecker",
    "LazyDict",
    "CustomFormatter",
    "flatten_dict",
    "build_json_encoder",
]

import json
import types
from string import Formatter
from collections import UserDict
from collections.abc import MutableMapping

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
    """Class for custom String formatting for more advanced String templates.
    
    - Provides support for inline string formatting (e.g. "{'Hi':10}")
    - Provides support for string template aligning by padding field names with spaces (e.g. "{some_field   :10}")
    - Provides support for custom formatting functions (e.g. "{some_list:len}" with format_funcs={"len": len})
    - Provides support for chained formatting (e.g. "{some_list:len:5}" with format_funcs={"len": len})
    - Provides support for elementwise formatting (e.g. "{some_list:@.2f:join}" with format_funcs={"join": ", ".join})
    - Provides support for default values for None values or missing keys
    """
    def __init__(self, format_funcs={}, default=None):
        super().__init__()
        self.format_funcs = format_funcs
        self.default = default
    
    def get_field(self, field_name, args, kwargs):
        field_name = field_name.strip()
        if (field_name.startswith("\'") and field_name.endswith("\'")) \
        or (field_name.startswith("\"") and field_name.endswith("\"")):
            # return inline string
            return field_name[1:-1], None
        else:
            try:
                # return field value
                return super().get_field(field_name, args, kwargs)
            except KeyError:
                if self.default is not None:
                    return None, None
                else:
                    raise
    
    def format_field(self, value, format_specs):
        format_specs = format_specs.split(":")
        # return default value for None values if provided
        if self.default is not None and value is None:
            # apply last format specification if it only specifies the width
            if len(format_specs) > 0 and format_specs[-1].isdecimal():
                return super(CustomFormatter, self).format_field(self.default, format_specs[-1])
            else:
                return self.default
        # iterate through pipeline of formatting specifications
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
