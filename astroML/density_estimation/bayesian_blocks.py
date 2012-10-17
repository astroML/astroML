"""
Bayesian Block implementation
=============================

Dynamic programming algorithm for finding the optimal adaptive-width histogram.

Based on Scargle 2012 [1]_

References
----------
[1] http://adsabs.harvard.edu/abs/2012arXiv1207.5578S
"""
#
# TODO: 
# - add more fitness functions.  The most important are events, binned counts
#   of events, and measurements with normal errors.
# - think about how to apply this to time-series rather than histogrammed data
#
import numpy as np

def nonbinned_fitness(N, T):
    """Non-Binned Event Fitness Function

    Parameters
    ----------
    N : float or array_like
        number of events in block
    T : float or array_like
        duration of block

    Returns
    -------
    logL : float or ndarray
        log-likelihood of the block

    See Also
    --------
    binned_fitness
    """
    return N * (np.log(N) - np.log(T))


def ncp_prior(N, p0):
    """Compute Prior on the number of changepoints

    This is done via the empirical formula in equation 21 of Scargle 2012

    Parameters
    ----------
    N : integer
        number of data points
    p0 : float
        false positive rate (0 <= p < 1)

    Returns
    -------
    prior : float
        prior on the number of changepoints
    """
    return 4 - 73.53 * p0 * (N ** -0.478)


def histogram_bayesian_blocks(t, fitness=nonbinned_fitness, p0=0.0):
    """Bayesian Blocks Implementation

    Parameters
    ----------
    t : array_like
        data to be histogrammed (one dimensional, length N)
    fitness : function (optional)
        fitness function to use.  The fitness function should accept two
        arguments:
 
        - an array of integers n (the number of points in each block)
        - an array of floats t (the width of each block)
 
        It should return the additive fitness for each block.  The default
        is the fitness function recommmended by Scargle 2012.
    p0 : float (optional)
        false-alarm probability.  This is used to compute the prior on
        the number of changepoints, using the empirical formula in eqn. 21
        of Scargle 2012.  Default is 0.00

    Returns
    -------
    counts : ndarray
        array containing counts in each of the N bins
    bins : ndarray
        array containing the (N+1) bin edges
    """
    t = np.array(t)
    assert t.ndim == 1

    t.sort()
    N = t.size

    # create length-(N + 1) array of cell edges
    edges = np.concatenate([t[:1],
                            0.5 * (t[1:] + t[:-1]),
                            t[-1:]])
    block_length = t[-1] - edges

    nn_vec = np.ones(N)

    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    #-----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    #-----------------------------------------------------------------
    for R in range(N):
        # Compute fit_vec : fitness of putative last block (end at R)
        width = block_length[:R + 1] - block_length[R + 1]
        width[np.where(width <= 0)] = np.inf

        nn_cum_vec = np.cumsum(nn_vec[:R + 1][::-1])[::-1]

        # evaluate fitness function
        fit_vec = fitness(nn_cum_vec, width)

        A_R = fit_vec - ncp_prior(R + 1, p0)
        A_R[1:] += best[:R]

        i_max = np.argmax(A_R)
        last[R] = i_max
        best[R] = A_R[i_max]
    
    #-----------------------------------------------------------------
    # Now find changepoints by iteratively peeling off the last block
    #-----------------------------------------------------------------
    change_points =  np.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]

    # compute counts and bins
    counts = np.diff(change_points)
    bins = edges[change_points]

    return counts, bins

if __name__ == '__main__':
    import pylab as pl
    np.random.seed(0)

    width = 1

    x = width * np.random.random(2000)
    x[0::20] *= 0.05
    x[0::20] += 0.25 * width

    x[1::20] *= 0.05
    x[1::20] += 0.50 * width

    x[2::20] *= 0.05
    x[2::20] += 0.75 * width

    ax = pl.subplot(211)
    ax.hist(x, bins=50, normed=True, alpha=0.5)

    ax = pl.subplot(212, sharex=ax)
    counts, bins = histogram_bayesian_blocks(x, p0=0.01)
    ax.hist(x, bins=bins, normed=True, alpha=0.5)

    ax.set_xlim(-0.1 * width, 1.1 * width)
    ax.set_ylim(0, 2.2)

    ax.xaxis.set_major_locator(pl.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(pl.MultipleLocator(0.05))
    pl.show()
