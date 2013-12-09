"""
Tools for density estimation

See also:
- sklearn.mixture.gmm : gaussian mixture models
- sklearn.neighbors.KernelDensity : Kernel Density Estimation (version 0.14+)
- astroML.density_estimation.XDGMM : extreme deconvolution
- scipy.spatial.gaussian_kde : a gaussian KDE implementation
"""
import warnings
import numpy as np
from scipy import special
from sklearn.metrics import pairwise_kernels, pairwise_distances
from sklearn.neighbors import BallTree

# TODO:
#       - KDE with errors (chp 6.1.2)


def n_volume(r, n):
    """compute the n-volume of a sphere of radius r in n dimensions"""
    return np.pi ** (0.5 * n) / special.gamma(0.5 * n + 1) * (r ** n)


class KDE(object):
    """Kernel Density Estimate

    .. note:: Deprecated in astroML 0.2
          Scikit-learn version 0.14 added a KernelDensity estimator class
          which has much better performance than this class.
          The ``KDE`` class will be removed in astroML version 0.3.

    Parameters
    ----------
    metric : string or callable
        ['gaussian'|'tophat'|'exponential']
        or one of the options in sklearn.metrics.pairwise_kernels.  See
        pairwise_kernels documentation for more information.
        For 'gaussian' or 'tophat', 'exponential', and 'quadratic',
        the results will be properly normalized in D dimensions.  This may
        not be the case for other metrics.

    h : float (optional)
        if metric is 'gaussian' or 'tophat', h gives the width of the kernel.
        Otherwise, h is not referenced.

    **kwargs :
        other keywords will be passed to the sklearn.metrics.pairwise_kernels
        function.

    Notes
    -----
    Kernel forms are as follows:

    - 'gaussian'    : K(x, y) ~ exp( -0.5 (x - y)^2 / h^2 )

    - 'tophat'      : K(x, y) ~ 1 if abs(x - y) < h
                              ~ 0 otherwise

    - 'exponential' : K(x, y) ~ exp(- abs(x - y) / h)

    - 'quadratic' : K(x, y) ~ (1 - (x - y)^2) if abs(x) < 1
                            ~ 0 otherwise

    All are properly normalized, so that their integral over all space is 1.

    See Also
    --------
    - sklearn.mixture.gmm : gaussian mixture models
    - KNeighborsDenstiy: nearest neighbors density estimation
    - scipy.spatial.gaussian_kde : a gaussian KDE implementation
    """
    def __init__(self, metric='gaussian', h=None, **kwargs):
        warnings.warn("The KDE class is deprecated as of version 0.2 and will "
                      "be removed in version 0.3. Upgrade to scikit-learn "
                      "version >0.14 and use sklearn.neighbors.KernelDensity "
                      "instead.")
        self.metric = metric
        self.kwargs = kwargs
        self.h = h
        self.factor = lambda ndim: 1

    def fit(self, X):
        """Train the kernel density estimator

        Parameters
        ----------
        X : array_like
            array of points to use to train the KDE.  Shape is
            (n_points, n_dim)
        """
        self.X_ = np.atleast_2d(X)
        if self.X_.ndim != 2:
            raise ValueError('X must be two-dimensional')
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

        if self.metric == 'gaussian':
            # wrangle gaussian into scikit-learn's 'rbf' kernel
            gamma = 0.5 / self.h / self.h
            D = pairwise_kernels(X, self.X_, metric='rbf', gamma=gamma)
            D /= np.sqrt(2 * np.pi * self.h ** (2 * X.shape[1]))
            dens = D.sum(1)

        elif self.metric == 'tophat':
            # use Ball Tree to efficiently count neighbors
            bt = BallTree(self.X_)
            counts = bt.query_radius(X, self.h,
                                     count_only=True)
            dens = counts / n_volume(self.h, X.shape[1])

        elif self.metric == 'exponential':
            D = pairwise_distances(X, self.X_)
            dens = np.exp(-abs(D) / self.h)
            dens = dens.sum(1)
            dens /= n_volume(self.h, X.shape[1]) * special.gamma(X.shape[1])

        elif self.metric == 'quadratic':
            D = pairwise_distances(X, self.X_)
            dens = (1 - (D / self.h) ** 2)
            dens[D > self.h] = 0
            dens = dens.sum(1)
            dens /= 2. * n_volume(self.h, X.shape[1]) / (X.shape[1] + 2)

        else:
            D = pairwise_kernels(X, self.X_, metric=self.metric, **self.kwargs)
            dens = D.sum(1)

        return dens


class KNeighborsDensity(object):
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
