import numpy as np
import pylab as pl
from .colortools import get_faded, colorWheel    

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
        ax = pl.gca()

    if filled:
        histtype = 'stepfilled'
    else:
        histtype = 'step'

    hist = ax.hist(data, bins, histtype=histtype,
                   lw=0, fc=c_fade, alpha=0.5, normed=normed,
                   label=label)
    plot = ax.plot(xfit, yfit, c=c)

    return hist, plot
