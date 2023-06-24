__all__ = ["MixtureDistribution"]

import numpy as np


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
