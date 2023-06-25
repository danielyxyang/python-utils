__all__ = ["case_distinction", "intersect_functions"]

import numpy as np


def case_distinction(cases, **kwargs):
    choices, conditions = zip(*cases)
    return np.select(conditions, choices, **kwargs)


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
    f1 = np.asarray(f1)
    f2 = np.asarray(f2)
    return np.nonzero(np.diff(np.sign(f1 - f2)))[0] + offset
