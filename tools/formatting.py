__all__ = [
    "format_size",
    "format_time",
    "CustomFormatter",
]

import re
from string import Formatter


# reference: https://github.com/pytorch/pytorch/blob/8bc04f46fe8e69188fa46f1611b46788a7d4824d/torch/cuda/memory.py#L494
def format_size(size, unit="auto", decimals=0):
    """Format byte size into human-readable format.

    Args:
        size (int): The number of bytes.
        unit (str, optional): The unit which should be used for formatting.
            Defaults to "auto".
        decimals (int, optional): The number of decimals which should be
            displayed. Defaults to 0.

    Returns:
        str: The byte size in human-readable format.
    """
    units = ["B ", "KB", "MB", "GB", "TB", "PB"]
    if unit not in ["auto"] + units:
        raise ValueError(f"Unit \"{unit}\" is not supported. Please provide one of {['auto'] + units}.")

    current_unit = units[0]
    for next_unit in units[1:]:
        if (unit == "auto" and size < 10 * 1000) or unit == current_unit:
            break
        current_unit = next_unit
        size /= 1000
    return f"{size:.{decimals}f}{current_unit}"


def format_time(time):
    """Format seconds into human-readable time format."""
    h, time = divmod(time, 3600)
    m, time = divmod(time, 60)
    s = time
    if h > 0:
        return f"{h:.0f}:{m:02.0f}h"
    elif m > 0:
        return f"{m:.0f}:{s:02.0f}min"
    elif s >= 10:
        return f"{s:.1f}s"
    elif s >= 1:
        return f"{s:.2f}s"
    else:
        return f"{s:.3f}s"


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

        self._str_format_spec_pattern = re.compile(r"^(.?[<>\^])?\d+$")

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
            if self._str_format_spec_pattern.match(format_specs[-1]):
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
