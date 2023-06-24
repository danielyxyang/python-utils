__all__ = ["cartesian_product", "setdiff2d", "is_in_range"]

import numpy as np


def cartesian_product(*arrays):
    """Compute N-dimensional cartesian product."""
    # https://stackoverflow.com/a/11146645
    arrays = np.asarray(arrays)
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)
    # return np.array(list(itertools.product(*arrays)))


def setdiff2d(a, b):
    """Compute set difference between two 2D lists."""
    # https://stackoverflow.com/a/11903368
    return np.setdiff1d(
        a.copy().view([("x", a.dtype), ("y", a.dtype)]),
        b.copy().view([("x", b.dtype), ("y", b.dtype)]),
    ).view(a.dtype).reshape(-1, 2)
    # a = set(map(tuple, a))
    # b = set(map(tuple, b))
    # return np.array(list(a.difference(b)))


def is_in_range(val, range, mod=None):
    """Check whether value is in the given range while supporting modulos."""
    min, max = range
    if mod is not None:
        min = min % mod
        val = val % mod
        max = max % mod
        if min > max:
            return np.logical_or(val <= max, min <= val)
    return np.logical_and(min <= val, val <= max)
