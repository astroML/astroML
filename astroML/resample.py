import numpy as np
from astroML.utils import check_random_state


def bootstrap(data, n_bootstraps, user_statistic, kwargs=None,
              pass_indices=False, random_state=None):
    """Compute bootstraped statistics of a dataset.

    Parameters
    ----------
    data : array_like
        A 1-dimensional data array of size n_samples
    n_bootstraps : integer
        the number of bootstrap samples to compute.  Note that internally,
        two arrays of size (n_bootstraps, n_samples) will be allocated.
        For very large numbers of bootstraps, this can cause memory issues.
    user_statistic : function
        The statistic to be computed.  This should take an array of data
        of size (n_bootstraps, n_samples) and return the row-wise statistics
        of the data.
    kwargs : dictionary (optional)
        A dictionary of keyword arguments to be passed to the
        user_statistic function.
    pass_indices : boolean (optional)
        if True, then the indices of the points rather than the points
        themselves are passed to `user_statistic`
    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    Returns
    -------
    distribution : ndarray
        the bootstrapped distribution of statistics (length = n_bootstraps)
    """
    # we don't set kwargs={} by default in the argument list, because using
    # a mutable type as a default argument can lead to strange results
    if kwargs is None:
        kwargs = {}

    rng = check_random_state(random_state)

    data = np.asarray(data)
    n_samples = data.size

    if data.ndim != 1:
        raise ValueError("bootstrap expects 1-dimensional data")

    # Generate random indices with repetition
    ind = rng.randint(n_samples, size=(n_bootstraps, n_samples))

    # Call the function
    if pass_indices:
        stat_bootstrap = user_statistic(ind, **kwargs)
    else:
        stat_bootstrap = user_statistic(data[ind], **kwargs)

    # compute the statistic on the data
    return stat_bootstrap


def jackknife(data, user_statistic, kwargs=None,
              return_raw_distribution=False, pass_indices=False):
    """Compute first-order jackknife statistics of the data.

    Parameters
    ----------
    data : array_like
        A 1-dimensional data array of size n_samples
    user_statistic : function
        The statistic to be computed.  This should take an array of data
        of size (n_samples, n_samples - 1) and return an array of size
        n_samples or tuple of arrays of size n_samples, representing the
        row-wise statistics of the input.
    kwargs : dictionary (optional)
        A dictionary of keyword arguments to be passed to the
        user_statistic function.
    return_raw_distribution : boolean (optional)
        if True, return the raw jackknife distribution.  Be aware that
        this distribution is not reflective of the true distribution:
        it is simply an intermediate step in the jackknife calculation
    pass_indices : boolean (optional)
        if True, then the indices of the points rather than the points
        themselves are passed to `user_statistic`

    Returns
    -------
    mean, stdev : floats
        The mean and standard deviation of the jackknifed distribution
    raw_distribution : ndarray
        Returned only if `return_raw_distribution` is True
        The array containing the raw distribution (length n_samples)
        Be aware that this distribution is not reflective of the true
        distribution: it is simply an intermediate step in the jackknife
        calculation

    Notes
    -----
    This implementation is a leave-one-out jackknife.
    Jackknife resampling is known to fail on rank-based statistics
    (e.g. median, quartiles, etc.)  It works well on smooth statistics
    (e.g. mean, standard deviation, etc.)
    """
    # we don't set kwargs={} by default in the argument list, because using
    # a mutable type as a default argument can lead to strange results
    if kwargs is None:
        kwargs = {}

    data = np.asarray(data)
    n_samples = data.size

    if data.ndim != 1:
        raise ValueError("bootstrap expects 1-dimensional data")

    # generate indices for the entire dataset, converting to row vector
    ind0 = np.arange(n_samples)[np.newaxis, :]

    # generate sets of indices where a single datapoint is left-out
    ind = np.arange(n_samples, dtype=int)
    ind = np.vstack([np.hstack((ind[:i], ind[i + 1:])) for i in ind])

    # compute the statistic for the whole dataset
    if pass_indices:
        stat_data = user_statistic(ind0, **kwargs)
        stat_jackknife = user_statistic(ind, **kwargs)
    else:
        stat_data = user_statistic(data[ind0], **kwargs)
        stat_jackknife = user_statistic(data[ind], **kwargs)

    # handle multiple statistics:
    # if ndim=0, then the statistic is not operating on rows (error).
    # if ndim=1, then it's a single statistic returned
    # if ndim=2, then a tuple has been returned
    stat_data = np.asarray(stat_data)
    ndim = stat_data.ndim

    if ndim == 0:
        raise ValueError("user_statistic should return row-wise statistics")

    stat_data = np.atleast_2d(stat_data).T
    stat_jackknife = np.atleast_2d(stat_jackknife)

    # compute the jackknife correction formula
    delta_stat = (n_samples - 1) * (stat_data - stat_jackknife.mean(1))
    stat_corrected = (stat_data + delta_stat)[0]
    sigma_stat = np.sqrt(1. / n_samples / (n_samples + 1)
                         * np.sum((n_samples * stat_data - stat_corrected
                                   - (n_samples - 1)
                                   * stat_jackknife.T) ** 2, 0))

    if return_raw_distribution:
        results = tuple(zip(stat_corrected, sigma_stat, stat_jackknife))
    else:
        results = tuple(zip(stat_corrected, sigma_stat))

    if ndim == 1:
        return results[0]
    else:
        return results
