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


class MixtureDistribution():
    def __init__(self, distributions):
        self.ps = np.array([p for p, _ in distributions])
        self.dists = [d for _, d in distributions]
    
    def sample(self, size=None):
        dist_i = np.random.choice(np.arange(len(self.dists)), size=size, p=self.ps)
        dist_rvs = np.array([dist.rvs(size) for dist in self.dists])
        samples = dist_rvs[dist_i, np.arange(size)]
        return samples
    
    def pdf(self, x):
        dist_pdfs = np.array([dist.pdf(x) for dist in self.dists])
        pdf = self.ps @ dist_pdfs
        return pdf
    
    def mean(self):
        dist_means = np.array([dist.mean() for dist in self.dists])
        mean = self.ps @ dist_means
        return mean
