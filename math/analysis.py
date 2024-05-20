import numpy as np


def case_distinction(cases, **kwargs):
    """Evaluate list of cases by case distinction using np.select.

    Args:
        cases (list of tuples): The list of tuples (value, condition) where
            condition is a boolean ndarray. When multiple conditions are
            satisfied, the first one encountered is used.

    Returns:
        list: The resulting list of values determined by case distinction.
    """
    choices, conditions = zip(*cases)
    return np.select(conditions, choices, **kwargs)


def intersect_functions(f1, f2, mode="left"):
    """Compute list of indices of the intersection points of two functions.

    The intersection points are computed by finding the zero-crossings of the
    difference of f1 and f2. Using "left" the point to the left of the
    intersection point is returned and using "right" the point to the right.

    Args:
        f1 (list): The list of values sampled from the first function.
        f2 (list): The list of values sampled from the second function.
        mode (str, optional): The mode "left" or "right" defining whether the
            index of the point to the left or right of the intersection should
            be returned. Defaults to "left".
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
