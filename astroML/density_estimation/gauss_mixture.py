import numpy as np
from sklearn.mixture import GaussianMixture


class GaussianMixture1D:
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

        components = data.shape[0]
        self._gmm = GaussianMixture(components, covariance_type='spherical')

        self._gmm.means_ = data[:, :1]
        self._gmm.weights_ = data[:, 2] / data[:, 2].sum()
        self._gmm.covariances_ = data[:, 1] ** 2

        self._gmm.precisions_cholesky_ = 1 / np.sqrt(self._gmm.covariances_)

        self._gmm.fit = None  # disable fit method for safety

    def sample(self, size):
        """Random sample"""
        return self._gmm.sample(size)

    def pdf(self, x):
        """Compute probability distribution"""

        if x.ndim == 1:
            x = x[:, np.newaxis]
        logprob = self._gmm.score_samples(x)
        return np.exp(logprob)

    def pdf_individual(self, x):
        """Compute probability distribution of each component"""

        if x.ndim == 1:
            x = x[:, np.newaxis]
        logprob = self._gmm.score_samples(x)
        responsibilities = self._gmm.predict_proba(x)
        return responsibilities * np.exp(logprob[:, np.newaxis])
