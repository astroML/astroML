import numpy as np
from matplotlib import pyplot as plt

from astroML.density_estimation import\
    scotts_bin_width, freedman_bin_width,\
    knuth_bin_width, bayesian_blocks


def hist(x, bins=10, range=None, *args, **kwargs):
    """Enhanced histogram

    This is a histogram function that enables the use of more sophisticated
    algorithms for determining bins.  Aside from the `bins` argument allowing
    a string specified how bins are computed, the parameters are the same
    as pylab.hist().

    Parameters
    ----------
    x : array_like
        array of data to be histogrammed

    bins : int or list or str (optional)
        If bins is a string, then it must be one of:
        'blocks' : use bayesian blocks for dynamic bin widths
        'knuth' : use Knuth's rule to determine bins
        'scotts' : use Scott's rule to determine bins
        'freedman' : use the Freedman-diaconis rule to determine bins

    range : tuple or None (optional)
        the minimum and maximum range for the histogram.  If not specified,
        it will be (x.min(), x.max())

    ax : Axes instance (optional)
        specify the Axes on which to draw the histogram.  If not specified,
        then the current active axes will be used.

    **kwargs :
        other keyword arguments are described in pylab.hist().
    """
    x = np.asarray(x)

    if 'ax' in kwargs:
        ax = kwargs['ax']
        del kwargs['ax']
    else:
        ax = plt.gca()

    # if range is specified, we need to truncate the data for
    # the bin-finding routines
    if (range is not None and (bins in ['blocks', 'knuth',
                                        'scotts', 'freedman'])):
        x = x[(x >= range[0]) & (x <= range[1])]

    if bins == 'blocks':
        bins = bayesian_blocks(x)
    elif bins == 'knuth':
        dx, bins = knuth_bin_width(x, True)
    elif bins == 'scotts':
        dx, bins = scotts_bin_width(x, True)
    elif bins == 'freedman':
        dx, bins = freedman_bin_width(x, True)
    elif isinstance(bins, str):
        raise ValueError("unrecognized bin code: '%s'" % bins)

    return ax.hist(x, bins, range, **kwargs)
