"""Routines to generate Probability Distribution Functions from sampled data."""

import numpy as np
from scipy import interpolate

class HistogramProbability:
    """Create a probability distribution function (pdf) using a histogram

    Parameters
    ----------
    data : array-like
        The 1D data for which the pdf will be generated

    bins : integer or array
        The bins parameter which will be passed to numpy.histogram
        If an integer, specifies the number of bins.
        If an array, specifies the bin edges.
        Default is ``bins = len(data) / 20``

    Examples
    --------
    >>> import numpy as np
    >>> from astroML.pdf import HistogramProbability
    >>> np.random.seed(0)
    >>> x = np.random.normal(size=1000)
    >>> hp = HistogramProbability(x)
    >>> hp(0)
    0.048120126389203162
    >>> hp(1)
    0.029262452371022096
    >>> hp(10)
    0.0

    See Also
    --------
    GaussianProbability
    """
    def __init__(self, data, bins=None):
        if bins is None:
            bins = len(data) / 20
        hist, bin_edges = np.histogram(data, bins)
        binsize = bin_edges[1] - bin_edges[0]

        # zero-pad the histogram: This assures that the pdf will
        # not be extrapolated
        self.hist_ = np.zeros(len(hist) + 4)
        self.hist_[2: -2] = hist * 1. / hist.sum()

        self.bins_ = np.zeros(len(hist) + 4)
        self.bins_[2: -2] = bin_edges[:-1] + 0.5 * binsize
        self.bins_[0] = self.bins_[2] - 2 * binsize
        self.bins_[1] = self.bins_[2] - binsize
        self.bins_[-2] = self.bins_[-3] + binsize
        self.bins_[-1] = self.bins_[-3] + 2 * binsize
        
        # Fit an order-1 spline to the distribution
        self.tck_ = interpolate.splrep(self.bins_, self.hist_, k=1)

    def __call__(self, x):
        return interpolate.splev(x, self.tck_)


class GaussianProbability:
    """Create a probability distribution function (pdf) using a gaussian fit

    Parameters
    ----------
    data : array-like
        The 1D data for which the pdf will be generated

    Attributes
    ----------
    mu : mean of the distribution
    sig2 : squared variance of the distribution

    Examples
    --------
    >>> import numpy as np
    >>> from astroML.pdf import GaussianProbability
    >>> np.random.seed(0)
    >>> x = np.random.normal(size=1000)
    >>> gp = GaussianProbability(x)
    >>> gp(0)
    0.40375861903377191
    >>> gp(1)
    0.23070506727149925
    >>> gp(10)
    1.3042586260824041e-23

    See Also
    --------
    HistogramProbability
    """
    def __init__(self, data):
        self.mu = np.mean(data)
        self.sig2 = np.mean((data - self.mu) ** 2)
        self.normalization = 1. / np.sqrt(2 * np.pi * self.sig2)

    def __call__(self, x):
        return (self.normalization
                * np.exp(-0.5 * (x - self.mu) ** 2 / self.sig2))
