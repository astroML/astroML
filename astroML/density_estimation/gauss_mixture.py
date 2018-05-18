import numpy as np
from sklearn.mixture import GaussianMixture


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

        precisions = [1/s**2 for s in sigmas]
        self._gmm = GaussianMixture(data.shape[0],
                                    weights_init=data[:, 2] / data[:, 2].sum(),
                                    means_init=data[:, :1],
                                    covariance_type='spherical',
                                    precisions_init=precisions)
        self._gmm.fit(data[:, :1])  # GaussianMixture requires 'fit' be called once
        self._gmm.fit = None  # disable fit method for safety

    def sample(self, size):
        """Random sample"""
        return self._gmm.sample(size)

    def pdf(self, x):
        """Compute probability distribution"""
        # logprob, responsibilities = self._gmm.eval(x)
        if x.ndim == 1:
            x = x[:, np.newaxis]
        logprob = self._gmm.score_samples(x)
        return np.exp(logprob)

    def pdf_individual(self, x):
        """Compute probability distribution of each component"""
        # logprob, responsibilities = self._gmm.eval(x)
        if x.ndim == 1:
            x = x[:, np.newaxis]
        logprob = self._gmm.score_samples(x)
        responsibilities = self._gmm.predict_proba(x)
        return responsibilities * np.exp(logprob[:, np.newaxis])
