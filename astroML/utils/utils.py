import numpy as np
from scipy import linalg
from sklearn.utils import check_random_state as sk_check_random_state
from astroML.utils.decorators import deprecated
from astroML.utils.exceptions import AstroMLDeprecationWarning

try:  # SciPy >= 0.19
    from scipy.special import logsumexp as scipy_logsumexp
except ImportError:
    from scipy.misc import logsumexp as scipy_logsumexp

__all__ = ['logsumexp', 'log_multivariate_gaussian', 'check_random_state',
           'split_samples', 'completeness_contamination', 'convert_2D_cov']


@deprecated('1.0', alternative='scipy.special.logsumexp',
            warning_type=AstroMLDeprecationWarning)
def logsumexp(arr, axis=None):
    return scipy_logsumexp(arr, axis)


def log_multivariate_gaussian(x, mu, V, Vinv=None, method=1):
    """Evaluate a multivariate gaussian N(x|mu, V)

    This allows for multiple evaluations at once, using array broadcasting

    Parameters
    ----------
    x: array_like
        points, shape[-1] = n_features

    mu: array_like
        centers, shape[-1] = n_features

    V: array_like
        covariances, shape[-2:] = (n_features, n_features)

    Vinv: array_like or None
        pre-computed inverses of V: should have the same shape as V

    method: integer, optional
        method = 0: use cholesky decompositions of V
        method = 1: use explicit inverse of V

    Returns
    -------
    values: ndarray
        shape = broadcast(x.shape[:-1], mu.shape[:-1], V.shape[:-2])

    Examples
    --------

    >>> x = [1, 2]
    >>> mu = [0, 0]
    >>> V = [[2, 1], [1, 2]]
    >>> log_multivariate_gaussian(x, mu, V)
    -3.3871832107434003
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    V = np.asarray(V, dtype=float)

    ndim = x.shape[-1]
    x_mu = x - mu

    if V.shape[-2:] != (ndim, ndim):
        raise ValueError("Shape of (x-mu) and V do not match")

    Vshape = V.shape
    V = V.reshape([-1, ndim, ndim])

    if Vinv is not None:
        assert Vinv.shape == Vshape
        method = 1

    if method == 0:
        Vchol = np.array([linalg.cholesky(V[i], lower=True)
                          for i in range(V.shape[0])])

        # we may be more efficient by using scipy.linalg.solve_triangular
        # with each cholesky decomposition
        VcholI = np.array([linalg.inv(Vchol[i])
                          for i in range(V.shape[0])])
        logdet = np.array([2 * np.sum(np.log(np.diagonal(Vchol[i])))
                           for i in range(V.shape[0])])

        VcholI = VcholI.reshape(Vshape)
        logdet = logdet.reshape(Vshape[:-2])

        VcIx = np.sum(VcholI * x_mu.reshape(x_mu.shape[:-1]
                                            + (1,) + x_mu.shape[-1:]), -1)
        xVIx = np.sum(VcIx ** 2, -1)

    elif method == 1:
        if Vinv is None:
            Vinv = np.array([linalg.inv(V[i])
                             for i in range(V.shape[0])]).reshape(Vshape)
        else:
            assert Vinv.shape == Vshape

        logdet = np.log(np.array([linalg.det(V[i])
                                  for i in range(V.shape[0])]))
        logdet = logdet.reshape(Vshape[:-2])

        xVI = np.sum(x_mu.reshape(x_mu.shape + (1,)) * Vinv, -2)
        xVIx = np.sum(xVI * x_mu, -1)

    else:
        raise ValueError("unrecognized method %s" % method)

    return -0.5 * ndim * np.log(2 * np.pi) - 0.5 * (logdet + xVIx)


@deprecated('1.0', alternative='sklearn.utils.check_random_state',
            warning_type=AstroMLDeprecationWarning)
def check_random_state(seed):
    return sk_check_random_state(seed)


def split_samples(X, y, fractions=[0.75, 0.25], random_state=None):
    """Split samples into training, test, and cross-validation sets

    Parameters
    ----------
    X, y : array_like
        leading dimension n_samples
    fraction : array_like
        length n_splits.  If the fractions do not add to 1, they will be
        re-normalized.
    random_state : None, int, or RandomState object
        random seed, or random number generator
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y should have the same leading dimension")

    n_samples = X.shape[0]

    fractions = np.asarray(fractions).ravel().cumsum()
    fractions /= fractions[-1]
    fractions *= n_samples
    N = np.concatenate([[0], fractions.astype(int)])
    N[-1] = n_samples  # in case of roundoff errors

    random_state = sk_check_random_state(random_state)
    indices = np.arange(len(y))
    random_state.shuffle(indices)

    X_divisions = tuple(X[indices[N[i]:N[i + 1]]]
                        for i in range(len(fractions)))
    y_divisions = tuple(y[indices[N[i]:N[i + 1]]]
                        for i in range(len(fractions)))

    return X_divisions, y_divisions


def completeness_contamination(predicted, true):
    """Compute the completeness and contamination values

    Parameters
    ----------
    predicted_value, true_value : array_like
        integer arrays of predicted and true values.  This assumes that
        'false' values are given by 0, and 'true' values are nonzero.

    Returns
    -------
    completeness, contamination : float or array_like
        the completeness and contamination of the results.  shape is
        np.broadcast(predicted, true).shape[:-1]
    """
    predicted = np.asarray(predicted)
    true = np.asarray(true)

    outshape = np.broadcast(predicted, true).shape[:-1]

    predicted = np.atleast_2d(predicted)
    true = np.atleast_2d(true)

    matches = (predicted == true)

    tp = np.sum(matches & (true != 0), -1)
    tn = np.sum(matches & (true == 0), -1)
    fp = np.sum(~matches & (true == 0), -1)
    fn = np.sum(~matches & (true != 0), -1)

    tot = (tp + fn)
    tot[tot == 0] = 1
    completeness = tp * 1. / tot

    tot = (tp + fp)
    tot[tot == 0] = 1
    contamination = fp * 1. / tot

    completeness[np.isnan(completeness)] = 0
    contamination[np.isnan(contamination)] = 0

    return completeness.reshape(outshape), contamination.reshape(outshape)


def convert_2D_cov(*args):
    """Convert a 2D covariance from matrix form to principal form, and back

    if one parameter is passed, it is a covariance matrix, and the principal
    axes and rotation (sigma1, sigma2, alpha) are returned.

    if three parameters are passed, they are assumed to be (sigma1, sigma2,
    alpha) and the covariance is returned
    """
    if len(args) == 1:
        C = np.asarray(args[0])
        if C.shape != (2, 2):
            raise ValueError("Input not understood")
        sigma_x2 = C[0, 0]
        sigma_y2 = C[1, 1]
        sigma_xy = C[0, 1]

        alpha = 0.5 * np.arctan2(2 * sigma_xy,
                                 (sigma_x2 - sigma_y2))
        tmp1 = 0.5 * (sigma_x2 + sigma_y2)
        tmp2 = np.sqrt(0.25 * (sigma_x2 - sigma_y2) ** 2 + sigma_xy ** 2)

        sigma1 = np.sqrt(tmp1 + tmp2)
        sigma2 = np.sqrt(tmp1 - tmp2)

        return (sigma1, sigma2, alpha)

    elif len(args) == 3:
        sigma1, sigma2, alpha = args

        s = np.sin(alpha)
        c = np.cos(alpha)
        sigma_x2 = (sigma1 * c) ** 2 + (sigma2 * s) ** 2
        sigma_y2 = (sigma1 * s) ** 2 + (sigma2 * c) ** 2
        sigma_xy = (sigma1 ** 2 - sigma2 ** 2) * s * c

        return np.array([[sigma_x2, sigma_xy],
                         [sigma_xy, sigma_y2]])

    else:
        raise ValueError("Input not understood")
