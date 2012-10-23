"""
Functions for regression using sums-of-norms
"""

import numpy as np


def norm(x, x0, sigma):
    return (1. / np.sqrt(2 * np.pi) / sigma
            * np.exp(-0.5 * (x - x0) ** 2 / sigma ** 2))


def sum_of_norms(x, y, num_gaussians=None, locs=None,
                 widths=None, spacing='linear', full_output=False):
    r"""Approximate a function with a sum of gaussians

    Parameters
    ----------
    x : array-like, shape = n_training
        The x-value of the input function
    y : array-like, shape = n_training
        The y-value of the input function
    num_gaussians : integer (optional)
        The number of gaussians to use.  If this is not specified, then the
        number of items in `locs` is used.  If neither is specified, this
        defaults to 30
    locs : array-like (optional)
        The locations of the gaussians to use.  If not specified, locations
        will be uniformly spaced between the end-points of x.
    widths : float or array-like (optional)
        The widths of the gaussians to use.  If a single value, use this for
        all widths.  If multiple values, the length must be equal to
        len(locs), if specified, and/or num_gaussians, if specified.
        If widths is not provided, then widths will be used which are
        half the distance between adjacent gaussians will be used.
    full_output : boolean (default = False)
        if True, return the rms error of the best-fit, the list of locations,
        and the list of widths
    spacing : string, ['linear'|'log']
        spacing to use for automatic determination of locs.  Not referenced
        if locs is specified

    Returns
    -------
    weights if full_output == False
    (weights, rms, locs, widths) if full_output == True

    weights : array-like, length = num_gaussians
        The weights which best approximate the spectrum.  The reconstruction
        is given by
        sum_{i=1}^{num_gaussians} weights[i] * norm(locs[i], widths[i])
    rms : float
        the root-mean-square error of the best-fit solution
    locs : array
        the locations of the gaussians used for the fit
    widths : array
        the widths of the gaussians used for the fit

    Notes
    -----
    This is solved using linear regression.  Our matrix :math:`X` has shape
    :math:`(m, n)` where :math:`m` is the number of training points, and
    :math:`n` is the number of gaussians in the fit.  We seek the linear
    combination of these :math:`n` gaussians which minimizes the squared
    residual error, which in matrix form can be expressed

    .. math:
        \epsilon = \min\left|y - Xw \right|

    here the vector :math:`w` encodes the linear combination.  The vector
    :math:`w` which minimizes :math:`\epsilon` can be shown to be

    .. math:
        w = (X^T X)^{-1} X^T y

    This is the result returned by this function.
    """
    x, y = map(np.asarray, (x, y))
    assert x.ndim == 1
    assert y.shape == x.shape

    n_training = x.shape[0]

    if locs is None:
        if num_gaussians is None:
            num_gaussians = 30
        if spacing == 'linear':
            locs = np.linspace(x[0], x[-1], num_gaussians)
        elif spacing == 'log':
            locs = np.logspace(np.log10(x[0]), np.log10(x[-1]), num_gaussians)
    else:
        locs = np.asarray(locs)
        if num_gaussians is None:
            num_gaussians = len(locs)
        if num_gaussians is not None:
            assert len(locs) == num_gaussians

    if widths is None:
        widths = np.zeros(num_gaussians)
        widths[:-1] = locs[1:] - locs[:-1]
        if len(widths) > 1:
            widths[-1] = widths[-2]
        else:
            widths[-1] = x[-1] - x[0]
    else:
        widths = np.atleast_1d(widths)
        assert widths.size in (1, num_gaussians)
        widths = widths + np.zeros(num_gaussians)  # broadcast to shape

    # use broadcasting to compute X in one go, without slow loops
    X = norm(x.reshape(n_training, 1),
             locs.reshape(1, num_gaussians),
             widths.reshape(1, num_gaussians))

    # use pinv rather than inv for numerical stability
    w_best = np.dot(np.linalg.pinv(np.dot(X.T, X)),
                    np.dot(X.T, y))

    if not full_output:
        return w_best
    else:
        rms = np.sqrt(np.mean(y - np.dot(X, w_best)) ** 2)
        return w_best, rms, locs, widths
