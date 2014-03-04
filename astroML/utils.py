import numpy as np
from scipy import linalg


try:
    # exists in python 2.7+
    from itertools import combinations_with_replacement
except:
    def combinations_with_replacement(iterable, r):
        """
        Return successive r-length combinations of elements in the iterable
        allowing individual elements to have successive repeats.
        combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC
        """
        from itertools import product
        pool = tuple(iterable)
        n = len(pool)
        for indices in product(range(n), repeat=r):
            if sorted(indices) == list(indices):
                yield tuple(pool[i] for i in indices)


def logsumexp(arr, axis=None):
    """Computes the sum of arr assuming arr is in the log domain.

    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow.

    Examples
    --------

    >>> import numpy as np
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsumexp(a)
    9.4586297444267107
    """
    # if axis is specified, roll axis to 0 so that broadcasting works below
    if axis is not None:
        arr = np.rollaxis(arr, axis)
        axis = 0

    # Use the max to normalize, as with the log this is what accumulates
    # the fewest errors
    vmax = arr.max(axis=axis)
    out = np.log(np.sum(np.exp(arr - vmax), axis=axis))
    out += vmax

    return out


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


# From scikit-learn utilities:
def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


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

    random_state = check_random_state(random_state)
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
