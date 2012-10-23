import numpy as np


def _sorted_interpolate(x, y, x_eval):
    """utility function for binned_Cminus"""
    # note that x should be sorted
    N = len(x)
    ind = x.searchsorted(x_eval)
    ind[ind == N] = N - 1

    y_eval = np.zeros(x_eval.shape)

    # find perfect matches
    match = (x[ind] == x_eval) | (x_eval > x[-1]) | (x_eval < x[0])
    y_eval[match] = y[ind[match]]

    ind = ind[~match]

    # take care of extrapolation
    ind[ind == 0] = 1

    x_lo = x[ind - 1]
    x_up = x[ind]

    y_lo = y[ind - 1]
    y_up = y[ind]

    # take care of places where x_lo = x_up

    y_eval[~match] = (y_lo + (x_eval[~match] - x_lo)
                      * (y_up - y_lo) / (x_up - x_lo))

    return y_eval


def Cminus(x, y, xmax, ymax):
    """Lynden-Bell's C-minus method

    Parameters
    ----------
    x : array_like
        array of x values
    y : array_like
        array of y values
    xmax : array_like
        array of maximum x values for each y value
    ymax : array_like
        array of maximum y values for each x value

    Returns
    -------
    Nx, Ny, cuml_x, cuml_y: ndarrays
        Nx and cuml_x are in the order of the sorted x array
        Ny and cuml_y are in the order of the sorted y array
    """
    # make copies of input
    x, y, xmax, ymax = map(np.array, (x, y, xmax, ymax))

    Nall = len(x)

    cuml_x = np.zeros(x.shape)
    cuml_y = np.zeros(y.shape)
    Nx = np.zeros(x.shape)
    Ny = np.zeros(y.shape)

    # first the y direction.
    i_sort = np.argsort(y)
    x = x[i_sort]
    y = y[i_sort]
    xmax = xmax[i_sort]
    ymax = ymax[i_sort]

    for j in range(1, Nall):
        Ny[j] = np.sum(x[:j] < xmax[j])
    Ny[0] = np.inf
    cuml_y = np.cumprod(1. + 1. / Ny)
    Ny[0] = 0

    # renormalize
    cuml_y *= Nall / cuml_y[-1]

    #now the x direction
    i_sort = np.argsort(x)
    x = x[i_sort]
    y = y[i_sort]
    xmax = xmax[i_sort]
    ymax = ymax[i_sort]

    for i in range(1, Nall):
        Nx[i] = np.sum(y[:i] < ymax[i])
    Nx[0] = np.inf
    cuml_x = np.cumprod(1. + 1. / Nx)
    Nx[0] = 0

    # renormalize
    cuml_x *= Nall / cuml_x[-1]

    return Nx, Ny, cuml_x, cuml_y


def binned_Cminus(x, y, xmax, ymax, xbins, ybins, normalize=False):
    """Compute the binned distributions using the Cminus method

    Parameters
    ----------
    x : array_like
        array of x values
    y : array_like
        array of y values
    xmax : array_like
        array of maximum x values for each y value
    ymax : array_like
        array of maximum y values for each x value
    xbins : array_like
        array of bin edges for the x function: size=Nbins_x + 1
    ybins : array_like
        array of bin edges for the y function: size=Nbins_y + 1
    normalize : boolean
        if true, then returned distributions are normalized.  Default
        is False.

    Returns
    -------
    dist_x, dist_y : ndarrays
        distributions of size Nbins_x and Nbins_y
    """
    Nx, Ny, cuml_x, cuml_y = Cminus(x, y, xmax, ymax)

    # simple linear interpolation using a binary search
    # interpolate the cumulative distributions
    x_sort = np.sort(x)
    y_sort = np.sort(y)

    Ix_edges = _sorted_interpolate(x_sort, cuml_x, xbins)
    Iy_edges = _sorted_interpolate(y_sort, cuml_y, ybins)

    if xbins[0] < x_sort[0]:
        Ix_edges[0] = cuml_x[0]
    if xbins[-1] > x_sort[-1]:
        Ix_edges[-1] = cuml_x[-1]

    if ybins[0] < y_sort[0]:
        Iy_edges[0] = cuml_y[0]
    if ybins[-1] > y_sort[-1]:
        Iy_edges[-1] = cuml_y[-1]

    x_dist = np.diff(Ix_edges) / np.diff(xbins)
    y_dist = np.diff(Iy_edges) / np.diff(ybins)

    if normalize:
        x_dist /= len(x)
        y_dist /= len(y)

    return x_dist, y_dist


def bootstrap_Cminus(x, y, xmax, ymax, xbins, ybins,
                     Nbootstraps=10, normalize=False):
    """
    Compute the binned distributions using the Cminus method, with
    bootstrapped estimates of the errors

    Parameters
    ----------
    x : array_like
        array of x values
    y : array_like
        array of y values
    xmax : array_like
        array of maximum x values for each y value
    ymax : array_like
        array of maximum y values for each x value
    xbins : array_like
        array of bin edges for the x function: size=Nbins_x + 1
    ybins : array_like
        array of bin edges for the y function: size=Nbins_y + 1
    Nbootstraps : int
        number of bootstrap resamplings to perform
    normalize : boolean
        if true, then returned distributions are normalized.  Default
        is False.

    Returns
    -------
    dist_x, err_x, dist_y, err_y : ndarrays
        distributions of size Nbins_x and Nbins_y
    """
    x, y, xmax, ymax = map(np.asarray, (x, y, xmax, ymax))

    x_dist = np.zeros((Nbootstraps, len(xbins) - 1))
    y_dist = np.zeros((Nbootstraps, len(ybins) - 1))

    for i in range(Nbootstraps):
        ind = np.random.randint(0, len(x), len(x))
        x_dist[i], y_dist[i] = binned_Cminus(x[ind], y[ind],
                                             xmax[ind], ymax[ind],
                                             xbins, ybins,
                                             normalize=normalize)

    return (x_dist.mean(0), x_dist.std(0, ddof=1),
            y_dist.mean(0), y_dist.std(0, ddof=1))
