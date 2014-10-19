"""
Extreme deconvolution solver

This follows Bovy et al.
http://arxiv.org/pdf/0905.2979v2.pdf

Arbitrary mixing matrices R are not yet implemented: currently, this only
works with R = I.
"""
from __future__ import print_function, division

from time import time

import numpy as np
from scipy import linalg

from sklearn.mixture import GMM
from ..utils import logsumexp, log_multivariate_gaussian, check_random_state


class XDGMM(object):
    """Extreme Deconvolution

    Fit an extreme deconvolution (XD) model to the data

    Parameters
    ----------
    n_components: integer
        number of gaussian components to fit to the data
    n_iter: integer (optional)
        number of EM iterations to perform (default=100)
    tol: float (optional)
        stopping criterion for EM iterations (default=1E-5)

    Notes
    -----
    This implementation follows Bovy et al. arXiv 0905.2979
    """
    def __init__(self, n_components, n_iter=100, tol=1E-5, verbose=False,
                 random_state = None):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

        # model parameters: these are set by the fit() method
        self.V = None
        self.mu = None
        self.alpha = None

    def fit(self, X, Xerr, R=None):
        """Fit the XD model to data

        Parameters
        ----------
        X: array_like
            Input data. shape = (n_samples, n_features)
        Xerr: array_like
            Error on input data.  shape = (n_samples, n_features, n_features)
        R : array_like
            (TODO: not implemented)
            Transformation matrix from underlying to observed data.  If
            unspecified, then it is assumed to be the identity matrix.
        """
        if R is not None:
            raise NotImplementedError("mixing matrix R is not yet implemented")

        X = np.asarray(X)
        Xerr = np.asarray(Xerr)
        n_samples, n_features = X.shape

        # assume full covariances of data
        assert Xerr.shape == (n_samples, n_features, n_features)

        # initialize components via a few steps of GMM
        # this doesn't take into account errors, but is a fast first-guess
        gmm = GMM(self.n_components, n_iter=10, covariance_type='full',
                  random_state=self.random_state).fit(X)
        self.mu = gmm.means_
        self.alpha = gmm.weights_
        self.V = gmm.covars_

        logL = self.logL(X, Xerr)

        for i in range(self.n_iter):
            t0 = time()
            self._EMstep(X, Xerr)
            logL_next = self.logL(X, Xerr)
            t1 = time()

            if self.verbose:
                print("%i: log(L) = %.5g" % (i + 1, logL_next))
                print("    (%.2g sec)" % (t1 - t0))

            if logL_next < logL + self.tol:
                break
            logL = logL_next

        return self

    def logprob_a(self, X, Xerr):
        """
        Evaluate the probability for a set of points

        Parameters
        ----------
        X: array_like
            Input data. shape = (n_samples, n_features)
        Xerr: array_like
            Error on input data.  shape = (n_samples, n_features, n_features)

        Returns
        -------
        p: ndarray
            Probabilities.  shape = (n_samples,)
        """
        X = np.asarray(X)
        Xerr = np.asarray(Xerr)
        n_samples, n_features = X.shape

        # assume full covariances of data
        assert Xerr.shape == (n_samples, n_features, n_features)

        X = X[:, np.newaxis, :]
        Xerr = Xerr[:, np.newaxis, :, :]
        T = Xerr + self.V

        return log_multivariate_gaussian(X, self.mu, T)

    def logL(self, X, Xerr):
        """Compute the log-likelihood of data given the model

        Parameters
        ----------
        X: array_like
            data, shape = (n_samples, n_features)
        Xerr: array_like
            errors, shape = (n_samples, n_features, n_features)

        Returns
        -------
        logL : float
            log-likelihood
        """
        return np.sum(logsumexp(self.logprob_a(X, Xerr), -1))

    def _EMstep(self, X, Xerr):
        """
        Perform the E-step (eq 16 of Bovy et al)
        """
        n_samples, n_features = X.shape

        X = X[:, np.newaxis, :]
        Xerr = Xerr[:, np.newaxis, :, :]

        w_m = X - self.mu

        T = Xerr + self.V

        #------------------------------------------------------------
        # compute inverse of each covariance matrix T
        Tshape = T.shape
        T = T.reshape([n_samples * self.n_components,
                       n_features, n_features])
        Tinv = np.array([linalg.inv(T[i])
                         for i in range(T.shape[0])]).reshape(Tshape)
        T = T.reshape(Tshape)

        #------------------------------------------------------------
        # evaluate each mixture at each point
        N = np.exp(log_multivariate_gaussian(X, self.mu, T, Vinv=Tinv))

        #------------------------------------------------------------
        # E-step:
        #  compute q_ij, b_ij, and B_ij
        q = (N * self.alpha) / np.dot(N, self.alpha)[:, None]

        tmp = np.sum(Tinv * w_m[:, :, np.newaxis, :], -1)
        b = self.mu + np.sum(self.V * tmp[:, :, np.newaxis, :], -1)

        tmp = np.sum(Tinv[:, :, :, :, np.newaxis]
                     * self.V[:, np.newaxis, :, :], -2)
        B = self.V - np.sum(self.V[:, :, :, np.newaxis]
                            * tmp[:, :, np.newaxis, :, :], -2)

        #------------------------------------------------------------
        # M-step:
        #  compute alpha, m, V
        qj = q.sum(0)

        self.alpha = qj / n_samples

        self.mu = np.sum(q[:, :, np.newaxis] * b, 0) / qj[:, np.newaxis]

        m_b = self.mu - b
        tmp = m_b[:, :, np.newaxis, :] * m_b[:, :, :, np.newaxis]
        tmp += B
        tmp *= q[:, :, np.newaxis, np.newaxis]
        self.V = tmp.sum(0) / qj[:, np.newaxis, np.newaxis]

    def sample(self, size=1, random_state=None):
        if random_state is None:
            random_state = self.random_state
        rng = check_random_state(random_state)
        shape = tuple(np.atleast_1d(size)) + (self.mu.shape[1],)
        npts = np.prod(size)

        alpha_cs = np.cumsum(self.alpha)
        r = np.atleast_1d(np.random.random(size))
        r.sort()

        ind = r.searchsorted(alpha_cs)
        ind = np.concatenate(([0], ind))
        if ind[-1] != size:
            ind[-1] = size

        draw = np.vstack([np.random.multivariate_normal(self.mu[i],
                                                        self.V[i],
                                                        (ind[i + 1] - ind[i],))
                          for i in range(len(self.alpha))])

        return draw.reshape(shape)
