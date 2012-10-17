import numpy as np
import pylab as pl

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_kernels


def _setup_bins(sample, bins, range):
    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        N, D = sample.shape

    nbin = np.empty(D, int)
    edges = D * [None]
    dedges = D * [None]

    try:
        M = len(bins)
        if M != D:
            raise AttributeError(
                    'The dimension of bins must be equal'\
                    ' to the dimension of the sample x.')
    except TypeError:
        # bins is an integer
        bins = D * [bins]

    # Select range for each dimension
    # Used only if number of bins is given.
    if range is None:
        # Handle empty input. Range can't be determined in that case, use 0-1.
        if N == 0:
            smin = np.zeros(D)
            smax = np.ones(D)
        else:
            smin = np.atleast_1d(np.array(sample.min(0), float))
            smax = np.atleast_1d(np.array(sample.max(0), float))
    else:
        smin = np.zeros(D)
        smax = np.zeros(D)
        for i in xrange(D):
            smin[i], smax[i] = range[i]
    # Make sure the bins have a finite width.
    for i in xrange(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - .5
            smax[i] = smax[i] + .5

    # Create edge arrays
    for i in xrange(D):
        if np.isscalar(bins[i]):
            if bins[i] < 1:
                raise ValueError("Element at index %s in `bins` should be "
                                 "a positive integer." % i)
            nbin[i] = bins[i] + 2 # +2 for outlier bins
            edges[i] = np.linspace(smin[i], smax[i], nbin[i]-1)
        else:
            edges[i] = np.asarray(bins[i], float)
            nbin[i] = len(edges[i])+1  # +1 for outlier bins
        dedges[i] = np.diff(edges[i])
        if np.any(np.asarray(dedges[i]) <= 0):
            raise ValueError("""
            Found bin edge of size <= 0. Did you specify `bins` with
            non-monotonic sequence?""")

    return sample, edges, N, D


def _make_grid(edges):
    D = len(edges)
    locs = [0.5 * (edge[:-1] + edge[1:]) for edge in edges]
    grid = np.empty([len(loc) for loc in locs] + [D])
    slices = [slice(0, len(loc)) for loc in locs]
    for i in xrange(D):
        grid[slices + [i]] = locs[i][[np.newaxis] * i
                                     + [slice(0,len(locs[i]))]
                                     + [np.newaxis] * (D - i - 1)]
    return grid


def knn_local_density(sample, bins=20, range=None, n_neighbors=30,
                      normalize_input_range=False):
    """Estimate the density at each point X using nearest neighbors

    Note that binning semantics here are identical to that used in
    np.histogramdd

    Parameters
    ----------
    sample : array_like, shape(N, D)
        sample of points whose density will be estimated. 
    bins : sequence or int, optional
        The bin specification in which density will be estimated:

        * A sequence of arrays describing the bin edges along each dimension.
        * The number of bins for each dimension (nx, ny, ... =bins)
        * The number of bins for all dimensions (nx=ny=...=bins).

    range : sequence, optional
        A sequence of lower and upper bin edges to be used if the edges are
        not given explicitely in `bins`. Defaults to the minimum and maximum
        values along each dimension.
    n_neighbors : int, optional
        number of neighbors to use for density estimation
    normalize_input_range : bool, optional
        if true, then normalize the range of input for KNN search

    Returns
    -------
    edges : sequence of arrays
        the edges of the bins used for each dimension
    densities : array
        the shape [n1, n2, ...nn] array of densities in each bin.
    """
    sample, edges, N, D = _setup_bins(sample, bins, range)

    # Handle empty input.
    if N == 0:
        return np.zeros(D), edges

    # normalize input
    extents = np.array([edge[-1] - edge[0] for edge in edges], dtype=float)
    if normalize_input_range:
        edges = [ed/ex for (ed, ex) in zip(edges, extents)]
        sample /= extents

    # make a grid of locations at the center of each pixel
    grid = _make_grid(edges)

    # find the nearest neighbors of the grid points
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree')
    nbrs.fit(sample)
    dist, ind = nbrs.kneighbors(grid)
    
    # compute un-normalized density from distances
    dist **= D
    density = 1. / dist.sum(-1)

    # normalize density: we need to multiply by the D-volume of each pixel
    widths = map(np.diff, edges)
    tot = density.copy()
    for i in xrange(D):
        tot *= np.diff(edges[i])[[np.newaxis] * i
                                 + [slice(0, len(edges[i]) - 1)]
                                 + [np.newaxis] * (D - i - 1)]

    # find number of samples which are in the grid range:
    Nin = np.logical_and.reduce([(sample[:, i] >= edges[i][0])
                                 & (sample[:, i] <= edges[i][-1])
                                 for i in xrange(len(edges))]).sum()

    density *= Nin / tot.sum()

    # un-normalize input
    if normalize_input_range:
        edges = [ed * ex for (ed, ex) in zip(edges, extents)]
        sample *= extents
        density /= np.prod(extents)

    return density, edges


def kernel_density_estimator(sample, bins=20, range=None, sigma=1):
    """Estimate the density at each point X using nearest neighbors

    Note that binning semantics here are identical to that used in
    np.histogramdd

    Parameters
    ----------
    sample : array_like, shape(N, D)
        sample of points whose density will be estimated. 
    bins : sequence or int, optional
        The bin specification in which density will be estimated:

        * A sequence of arrays describing the bin edges along each dimension.
        * The number of bins for each dimension (nx, ny, ... =bins)
        * The number of bins for all dimensions (nx=ny=...=bins).

    range : sequence, optional
        A sequence of lower and upper bin edges to be used if the edges are
        not given explicitely in `bins`. Defaults to the minimum and maximum
        values along each dimension.
    sigma : scalar or array_like
        range over which to smooth data

    Returns
    -------
    edges : sequence of arrays
        the edges of the bins used for each dimension
    densities : array
        the shape [n1, n2, ...nn] array of densities in each bin.
    """
    sample, edges, N, D = _setup_bins(sample, bins, range)

    # Handle empty input.
    if N == 0:
        return np.zeros(D), edges

    # Check value of sigma
    sigma = np.asarray(sigma)
    if sigma.size not in (1, D):
        raise ValueError('size of sigma must be 1 or D')
    elif np.all(sigma == sigma.flat[0]):
        un_norm = None
        gamma = 0.5 / sigma.flat[0] ** 2
    else:
        # sigmas unequal: scale data
        for e, s in zip(edges, sigma.flat):
            e /= s ** 2
        sample /= sigma.ravel() ** 2
        gamma = 0.5
        un_norm = sigma.ravel() ** 2

    # make a grid of locations at the center of each pixel
    grid = _make_grid(edges)

    # compute distance grid
    K = pairwise_kernels(grid.reshape(-1, D), sample, metric='rbf', gamma=gamma)

    density = K.sum(-1)

    density = density.reshape(grid.shape[:-1])

    # normalize density: we need to multiply by the D-volume of each pixel
    widths = map(np.diff, edges)
    tot = density.copy()
    for i in xrange(D):
        tot *= np.diff(edges[i])[[np.newaxis] * i
                                 + [slice(0, len(edges[i]) - 1)]
                                 + [np.newaxis] * (D - i - 1)]

    # find number of samples which are in the grid range:
    Nin = np.logical_and.reduce([(sample[:, i] >= edges[i][0])
                                 & (sample[:, i] <= edges[i][-1])
                                 for i in xrange(len(edges))]).sum()

    density *= Nin / tot.sum()

    # un-normalize input
    if un_norm is not None:
        edges = [ed * n for (ed, n) in zip(edges, un_norm)]
        sample *= un_norm
        density /= np.prod(un_norm)

    return density, edges

