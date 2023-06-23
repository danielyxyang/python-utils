import numpy as np

def setdiff2d(a, b):
    """Compute set difference between two 2D lists."""
    return np.setdiff1d(
        a.copy().view([("x", a.dtype), ("y", a.dtype)]),
        b.copy().view([("x", a.dtype), ("y", a.dtype)]),
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


def intersect_functions(f1, f2, mode="left"):
    """Compute list of indices of the intersection points of two functions.
    
    The intersection points are computed by finding the zero-crossings of the
    difference of f1 and f2. Using "left" the point to the left of the
    intersection point is returned and using "right" the point to the right.
    """
    if mode == "left":
        offset = 0
    elif mode == "right":
        offset = 1
    else:
        print("WARNING: unknown mode for intersecting two functions")
    return np.nonzero(np.diff(np.sign(f1 - f2)))[0] + offset