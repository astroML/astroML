import numpy as np
from matplotlib import pyplot as plt
from astroML.plotting.colortools import get_faded, colorWheel

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
    
    other keyword arguments are described in pylab.hist().
    """
    # TODO: use an event-based Bayesian Blocks fitness function to allow for
    #       data with repeated values (see ValueError below)
    x = np.asarray(x)

    if 'ax' in kwargs:
        ax = kwargs['ax']
        del kwargs['ax']
    else:
        ax = plt.gca()

    # if range is specified, we need to truncate the data for
    # the bin-finding routines
    if (range is not None and
             (bins in ['blocks', 'knuth', 'scotts', 'freedman'])):
        x = x[(x >= range[0]) & (x <= range[1])]

    if bins == 'blocks':
        unique = np.unique(x)
        if unique.size < x.size:
            raise ValueError("bins='blocks' does not yet support data "
                             "with repeated values")
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


def hist_with_fit(data, xfit, yfit, bins=None, c=None, fade=0.5, normed=True,
                  filled=True, ax=None, label=None):
    """Plot a histogram with a line of best-fit.

    The color of the histogram and the line will match, and the histogram is
    transparent so that multiple datasets can be shown together on the same
    plot.

    Parameters
    ----------
    data: array-like
        data for the histogram
    xfit: array-like
        x values for the fit line
    yfit: array-like
        y values for the fit line

    Other Parameters
    ----------------
    bins: integer, or array
        bins which are passed to the histogram function
    c: string
        color of the line
    fade: float, 0 < fade <= 1
        amount by which histogram color is faded in comparison to the line
    normed: boolean (default=True)
        if True, normalize the histogram plot
    filled: boolean (default=True)
        if True, fill the histogram with a solid color
    ax: matplotlib axes instance
        if not specified, the current axes will be used
    label: string, the label of the data & fit
        label will be passed to the histogram function
      
    Returns
    -------
    hist:
        the tuple returned from matplotlib.pyplot.hist()
    plot:
        the typle returned from matplotlib.pyplot.plot()

    """
    (data, xfit, yfit) = map(np.asarray, (data, xfit, yfit))

    if bins is None:
        bins = len(data) / 20
    if c is None:
        c = colorWheel.next()

    c_fade = get_faded(c, fade)

    if ax is None:
        ax = plt.gca()

    if filled:
        histtype = 'stepfilled'
    else:
        histtype = 'step'

    hist = hist(data, bins, ax=ax, histtype=histtype,
                lw=0, fc=c_fade, alpha=0.5, normed=normed,
                label=label)
    plot = ax.plot(xfit, yfit, c=c)

    return hist, plot
