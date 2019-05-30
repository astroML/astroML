"""
Tools for density estimation

See also:
- sklearn.mixture.gmm : gaussian mixture models
- sklearn.neighbors.KernelDensity : Kernel Density Estimation (version 0.14+)
- astroML.density_estimation.XDGMM : extreme deconvolution
- scipy.spatial.gaussian_kde : a gaussian KDE implementation
"""
import numpy as np
from scipy import special
from sklearn.neighbors import BallTree


def n_volume(r, n):
    """compute the n-volume of a sphere of radius r in n dimensions"""
    return np.pi ** (0.5 * n) / special.gamma(0.5 * n + 1) * (r ** n)


class KNeighborsDensity:
    """K-neighbors density estimation

    Parameters
    ----------
    method : string
        method to use.  Must be one of ['simple'|'bayesian'] (see below)
    n_neighbors : int
        number of neighbors to use

    Notes
    -----
    The two methods are as follows:

    - simple:
        The density at a point x is estimated by n(x) ~ k / r_k^n
    - bayesian:
        The density at a point x is estimated by n(x) ~ sum_{i=1}^k[1 / r_i^n].

    See Also
    --------
    KDE : kernel density estimation
    """
    def __init__(self, method='bayesian', n_neighbors=10):
        if method not in ['simple', 'bayesian']:
            raise ValueError("method = %s not recognized" % method)

        self.n_neighbors = n_neighbors
        self.method = method

    def fit(self, X):
        """Train the K-neighbors density estimator

        Parameters
        ----------
        X : array_like
            array of points to use to train the KDE.  Shape is
            (n_points, n_dim)
        """
        self.X_ = np.atleast_2d(X)

        if self.X_.ndim != 2:
            raise ValueError('X must be two-dimensional')

        self.bt_ = BallTree(self.X_)

        return self

    def eval(self, X):
        """Evaluate the kernel density estimation

        Parameters
        ----------
        X : array_like
            array of points at which to evaluate the KDE.  Shape is
            (n_points, n_dim), where n_dim matches the dimension of
            the training points.

        Returns
        -------
        dens : ndarray
            array of shape (n_points,) giving the density at each point.
            The density will be normalized for metric='gaussian' or
            metric='tophat', and will be unnormalized otherwise.
        """
        X = np.atleast_2d(X)
        if X.ndim != 2:
            raise ValueError('X must be two-dimensional')

        if X.shape[1] != self.X_.shape[1]:
            raise ValueError('dimensions of X do not match training dimension')

        dist, ind = self.bt_.query(X, self.n_neighbors, return_distance=True)

        k = float(self.n_neighbors)
        ndim = X.shape[1]

        if self.method == 'simple':
            return k / n_volume(dist[:, -1], ndim)

        elif self.method == 'bayesian':
            # XXX this may be wrong in more than 1 dimension!
            return (k * (k + 1) * 0.5 / n_volume(1, ndim)
                    / (dist ** ndim).sum(1))
        else:
            raise ValueError("Unrecognized method '%s'" % self.method)

        return dens
