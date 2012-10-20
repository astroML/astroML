"""
Bayesian Block implementation
=============================

Dynamic programming algorithm for finding the optimal adaptive-width histogram.

Based on Scargle 2012 [1]_

References
----------
[1] http://adsabs.harvard.edu/abs/2012arXiv1207.5578S
"""
import numpy as np

# Fitness Functions
# ----------------- 
# Take the following parameters (can be arrays):
#  T : width of block
#  N : number of counts in block
#  a_k: 0.5 sum[1 / sigma_n^2]
#  b_k: - sum[x_n / sigma_n^2]
#  c_k: 0.5 sum[x_n^2 / sigma_n^2]

class FitnessFunc:
    """Base class for fitness functions

    Each fitness function class has the following:

    args attribute : a list of the arguments computed for the fitness function
    fitness(**kwargs) : compute fitness function from arguments in args
    prior(N) : compute prior on N
    """
    args = []
    def __init__(self):
        raise NotImplementedError()

    def fitness(**kwargs):
        raise NotImplementedError()

    def prior(N):
        raise NotImplementedError()


class EventsFitness(FitnessFunc):
    """Fitness for binned or unbinned events

    Parameters
    ----------
    p0 : float
        False alarm probability, used to compute the prior on N
        (see eq. 21 of Scargle 2012)
    """
    args = ['N_k', 'T_k']
    def __init__(self, p0=0):
        self.p0 = 0

    def fitness(self, N_k, T_k):
        # eq. 19 from Scargle 2012
        return N_k * (np.log(N_k) - np.log(T_k))

    def prior(self, N):
        # eq. 21 from Scargle 2012
        return 4 - 73.53 * self.p0 * (N ** -0.478)


class MeasuresFitness(FitnessFunc):
    """Fitness for point measures"""
    args = ['a_k', 'b_k']
    def __init__(self):
        pass

    def fitness(self, a_k, b_k):
        # eq. 19 from Scargle 2012
        return (b_k * b_k) / (4 * a_k)

    def prior(self, N):
        # eq. at end of sec 3.2 in Scargle 2012
        return 1.32 + 0.577 * np.log10(N)


def bayesian_blocks(t, x=None, sigma=None,
                    fitness='events', **kwargs):
    """Bayesian Blocks Implementation

    Parameters
    ----------
    t : array_like
        data times (one dimensional, length N)
    x : array_like (optional)
        data values
    sigma : array_like or float (optional)
        data errors
    fitness : str or object
        the fitness function to use.
        If a string, the following options are supported:

        - 'events' : fitness for binned or unbinned event data
        - 'measures' : fitness for a measured sequence with Gaussian errors

        In these cases, any additional keyword arguments are passed to the
        fitness function constructor.
        Alternatively, the fitness can be a user-specified object of
        type derived from the FitnessFunc class.

    Returns
    -------
    edges : ndarray
        array containing the (N+1) bin edges

    See Also
    --------
    astroML.plotting.hist
    """
    # validate array input
    t = np.asarray(t, dtype=float)
    if x is not None:
        x = np.asarray(x)
    if sigma is not None:
        sigma = np.asarray(sigma)

    # verify the fitness function
    if fitness == 'events':
        if x is not None and np.any(x % 1 > 0):
            raise ValueError("x must be integer counts for fitness='events'")
        fitfunc = EventsFitness(**kwargs)
    elif fitness == 'measures':
        if x is None:
            raise ValueError("x must be specified for fitness='measures'")
        fitfunc = MeasuresFitness(**kwargs)
    else:
        if not (hasattr(fitness, 'args') and
                hasattr(fitness, 'fitness') and
                hasattr(fitness, 'prior')):
            raise ValueError("fitness not understood")
        fitfunc = fitness

    # find unique values of t
    t = np.array(t, dtype=float)
    assert t.ndim == 1
    unq_t, unq_ind, unq_inv = np.unique(t, return_index=True,
                                        return_inverse=True)

    # if x is not specified, x will be counts at each time
    if x is None:
        if sigma is not None:
            raise ValueError("If sigma is specified, x must be specified")

        if len(unq_t) == len(t):
            x = np.ones_like(t)
        else:
            x = np.bincount(unq_inv)

        t = unq_t
        sigma = 1

    # if x is specified, then we need to sort t and x together
    else:
        x = np.asarray(x)

        if len(t) != len(x):
            raise ValueError("Size of t and x does not match")

        if len(unq_t) != len(t):
            raise ValueError("Repeated values in t not supported when "
                             "x is specified")
        
        t = unq_t
        x = x[unq_ind]

    # verify the given sigma value
    N = t.size
    if sigma is not None:
        sigma = np.asarray(sigma)
        if sigma.shape not in [(), (1,), (N,)]:
            raise ValueError('sigma does not match the shape of x')
    else:
        sigma = 1

    # compute values needed for computation, below
    ak_raw = np.ones_like(x) / sigma / sigma
    bk_raw = x * ak_raw
    ck_raw = x * bk_raw

    # create length-(N + 1) array of cell edges
    edges = np.concatenate([t[:1],
                            0.5 * (t[1:] + t[:-1]),
                            t[-1:]])
    block_length = t[-1] - edges

    # arrays to store the best configuration
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    #-----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    #-----------------------------------------------------------------
    for R in range(N):
        # Compute fit_vec : fitness of putative last block (end at R)
        kwds = {}
        
        # T_k: width/duration of each block
        if 'T_k' in fitfunc.args:
            kwds['T_k'] = block_length[:R + 1] - block_length[R + 1]

        # N_k: number of elements in each block
        if 'N_k' in fitfunc.args:
            kwds['N_k'] = np.cumsum(x[:R + 1][::-1])[::-1]

        # a_k: eq. 31
        if 'a_k' in fitfunc.args:
            kwds['a_k'] = 0.5 * np.cumsum(ak_raw[:R + 1][::-1])[::-1]

        # b_k: eq. 32
        if 'b_k' in fitfunc.args:
            kwds['b_k'] = - np.cumsum(bk_raw[:R + 1][::-1])[::-1]

        # c_k: eq. 33
        if 'c_k' in fitfunc.args:
            kwds['c_k'] = 0.5 * np.cumsum(ck_raw[:R + 1][::-1])[::-1]

        # evaluate fitness function
        fit_vec = fitfunc.fitness(**kwds)

        A_R = fit_vec - fitfunc.prior(R + 1)
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

    return edges[change_points]
