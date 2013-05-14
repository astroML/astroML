import numpy as np
from scipy import stats

#from scipy.special import erfinv
#sigmaG_factor = 1. / (2 * np.sqrt(2) * erfinv(0.5))
sigmaG_factor = 0.74130110925280102


def mean_sigma(a, axis=None, dtype=None, ddof=0, keepdims=False):
    """Compute mean and standard deviation for an array

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : int, optional
        Axis along which the means are computed. The default is to compute
        the mean of the flattened array.
    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of
        integer type the default is float64, for arrays of float types it is
        the same as the array type.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    mu : ndarray, see dtype parameter above
        array containing the mean values

    sigma : ndarray, see dtype parameter above.
        array containing the standard deviation

    See Also
    --------
    median_sigmaG : robust rank-based version of this calculation.

    Notes
    -----
    This routine simply calls ``np.mean`` and ``np.std``, passing the
    keyword arguments to them.  It is provided for ease of comparison
    with the function median_sigmaG()
    """
    mu = np.mean(a, axis=axis, dtype=dtype)
    sigma = np.std(a, axis=axis, dtype=dtype, ddof=ddof)

    if keepdims:
        if axis is None:
            newshape = a.ndim * (1,)
        else:
            newshape = np.asarray(a.shape)
            newshape[axis] = 1

        mu = mu.reshape(newshape)
        sigma = sigma.reshape(newshape)

    return mu, sigma


def median_sigmaG(a, axis=None, overwrite_input=False, keepdims=False):
    """Compute median and rank-based estimate of the standard deviation

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : int, optional
        Axis along which the means are computed. The default is to compute
        the mean of the flattened array.
    overwrite_input : bool, optional
       If True, then allow use of memory of input array `a` for
       calculations. The input array will be modified by the call to
       median. This will save memory when you do not need to preserve
       the contents of the input array. Treat the input as undefined,
       but it will probably be fully or partially sorted.
       Default is False. Note that, if `overwrite_input` is True and the
       input is not already an array, an error will be raised.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    median : ndarray, see dtype parameter above
        array containing the median values

    sigmaG : ndarray, see dtype parameter above.
        array containing the robust estimator of the standard deviation

    See Also
    --------
    mean_sigma : non-robust version of this calculation
    sigmaG : robust rank-based estimate of standard deviation

    Notes
    -----
    This routine uses a single call to ``np.percentile`` to find the
    quartiles along the given axis, and uses these to compute the
    median and sigmaG:

    median = q50
    sigmaG = (q75 - q25) * 0.7413

    where 0.7413 ~ 1 / (2 sqrt(2) erf^-1(0.5))
    """
    q25, median, q75 = np.percentile(a, [25, 50, 75],
                                     axis=axis,
                                     overwrite_input=overwrite_input)
    sigmaG = sigmaG_factor * (q75 - q25)

    if keepdims:
        if axis is None:
            newshape = a.ndim * (1,)
        else:
            newshape = np.asarray(a.shape)
            newshape[axis] = 1

        median = median.reshape(newshape)
        sigmaG = sigmaG.reshape(newshape)

    return median, sigmaG


def sigmaG(a, axis=None, overwrite_input=False, keepdims=False):
    """Compute the rank-based estimate of the standard deviation

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : int, optional
        Axis along which the means are computed. The default is to compute
        the mean of the flattened array.
    overwrite_input : bool, optional
       If True, then allow use of memory of input array `a` for
       calculations. The input array will be modified by the call to
       median. This will save memory when you do not need to preserve
       the contents of the input array. Treat the input as undefined,
       but it will probably be fully or partially sorted.
       Default is False. Note that, if `overwrite_input` is True and the
       input is not already an array, an error will be raised.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    median : ndarray, see dtype parameter above
        array containing the median values

    sigmaG : ndarray, see dtype parameter above.
        array containing the robust estimator of the standard deviation

    See Also
    --------
    median_sigmaG : robust rank-based estimate of mean and standard deviation

    Notes
    -----
    This routine uses a single call to ``np.percentile`` to find the
    quartiles along the given axis, and uses these to compute the
    sigmaG, a robust estimate of the standard deviation sigma:

    sigmaG = 0.7413 * (q75 - q25)

    where 0.7413 ~ 1 / (2 sqrt(2) erf^-1(0.5))
    """
    q25, q75 = np.percentile(a, [25, 75],
                             axis=axis,
                             overwrite_input=overwrite_input)
    sigmaG = sigmaG_factor * (q75 - q25)

    if keepdims:
        if axis is None:
            newshape = a.ndim * (1,)
        else:
            newshape = np.asarray(a.shape)
            newshape[axis] = 1

        sigmaG = sigmaG.reshape(newshape)

    return sigmaG


def fit_bivariate_normal(x, y, robust=False):
    """Fit bivariate normal parameters to a 2D distribution of points

    Parameters
    ----------
    x, y : array_like
        The x, y coordinates of the points

    robust : boolean (optional, default=False)
        If True, then use rank-based statistics which are robust to outliers
        Otherwise, use mean/std statistics which are not robust

    Returns
    -------
    mu : tuple
        (x, y) location of the best-fit bivariate normal
    sigma_1, sigma_2 : float
        The best-fit gaussian widths in the uncorrelated frame
    alpha : float
        The rotation angle in radians of the uncorrelated frame
    """
    x = np.asarray(x)
    y = np.asarray(y)

    assert x.shape == y.shape

    if robust:
        # use quartiles to compute center and spread
        med_x, sigmaG_x = median_sigmaG(x)
        med_y, sigmaG_y = median_sigmaG(y)

        # define the principal variables from Shevlyakov & Smirnov (2011)
        sx = 2 * sigmaG_x
        sy = 2 * sigmaG_y

        u = (x / sx + y / sy) / np.sqrt(2)
        v = (x / sx - y / sy) / np.sqrt(2)

        med_u, sigmaG_u = median_sigmaG(u)
        med_v, sigmaG_v = median_sigmaG(v)

        r_xy = ((sigmaG_u ** 2 - sigmaG_v ** 2) /
                (sigmaG_u ** 2 + sigmaG_v ** 2))

        # rename estimators
        mu_x, mu_y = med_x, med_y
        sigma_x, sigma_y = sigmaG_x, sigmaG_y
    else:
        mu_x = np.mean(x)
        sigma_x = np.std(x)

        mu_y = np.mean(y)
        sigma_y = np.std(y)

        r_xy = stats.pearsonr(x, y)[0]

    # We need to use the full (-180, 180) version of arctan: this is
    # np.arctan2(x, y) = np.arctan(x / y), modulo 180 degrees
    sigma_xy = r_xy * sigma_x * sigma_y
    alpha = 0.5 * np.arctan2(2 * sigma_xy, sigma_x ** 2 - sigma_y ** 2)

    sigma1 = np.sqrt((0.5 * (sigma_x ** 2 + sigma_y ** 2)
                      + np.sqrt(0.25 * (sigma_x ** 2 - sigma_y ** 2) ** 2
                                + sigma_xy ** 2)))
    sigma2 = np.sqrt((0.5 * (sigma_x ** 2 + sigma_y ** 2)
                      - np.sqrt(0.25 * (sigma_x ** 2 - sigma_y ** 2) ** 2
                                + sigma_xy ** 2)))

    return [mu_x, mu_y], sigma1, sigma2, alpha
