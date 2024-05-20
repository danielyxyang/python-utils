import numpy as np


class MixtureDistribution():
    """Class defining a mixture distribution."""

    def __init__(self, distributions):
        """Create a mixture distribution.

        Args:
            distributions (list of tuples): The list of tuples (p, dist) with p
                defining the mixture probability of the mixture component dist.
                The distribution dist must be of type `scipy.stats.rv_generic`.
        """
        self.ps = np.array([p for p, _ in distributions])
        self.dists = [d for _, d in distributions]

    def sample(self, size=None):
        """Sample values from the mixture distribution."""
        dist_i = np.random.choice(np.arange(len(self.dists)), size=size, p=self.ps)
        dist_rvs = np.array([dist.rvs(size) for dist in self.dists])
        samples = dist_rvs[dist_i, np.arange(size)]
        return samples

    def pdf(self, x):
        """Evaluate the density function of the mixture distribution."""
        dist_pdfs = np.array([dist.pdf(x) for dist in self.dists])
        pdf = self.ps @ dist_pdfs
        return pdf

    def mean(self):
        """Evaluate the mean of the mixtrue distribution."""
        dist_means = np.array([dist.mean() for dist in self.dists])
        mean = self.ps @ dist_means
        return mean
