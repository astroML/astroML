import numpy as np
from sklearn.mixture import GMM


class GaussianMixture1D(object):
    """
    Simple class to work with 1D mixtures of Gaussians

    Parameters
    ----------
    means : array_like
        means of component distributions (default = 0)
    sigmas : array_like
        standard deviations of component distributions (default = 1)
    weights : array_like
        weight of component distributions (default = 1)
    """
    def __init__(self, means=0, sigmas=1, weights=1):
        data = np.array([t for t in np.broadcast(means, sigmas, weights)])

        self._gmm = GMM(data.shape[0])
        self._gmm.fit = None  # disable fit method for safety

        self._gmm.means_ = data[:, :1]
        self._gmm.covars_ = data[:, 1:2] ** 2
        self._gmm.weights = data[:, 2] / data[:, 2].sum()

    def sample(self, size):
        """Random sample"""
        return self._gmm.sample(size)

    def pdf(self, x):
        """Compute probability distribution"""
        logprob, responsibilities = self._gmm.eval(x)
        return np.exp(logprob)

    def pdf_individual(self, x):
        """Compute probability distribution of each component"""
        logprob, responsibilities = self._gmm.eval(x)
        return responsibilities * np.exp(logprob[:, np.newaxis])
