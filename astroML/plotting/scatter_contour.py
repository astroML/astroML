import numpy as np

class ShapeError(ValueError):
    pass


def scatter_contour(x, y,
                    levels=10,
                    threshold=100,
                    log_counts=False,
                    histogram2d_args=None,
                    plot_args=None,
                    contour_args=None,
                    filled_contour=True,
                    xerr=None,
                    yerr=None,
                    ax=None):
    """Scatter plot with contour over dense regions

    Parameters
    ----------
    x, y : arrays
        x and y data for the contour plot
    levels : integer or array (optional, default=10)
        number of contour levels, or array of contour levels
    threshold : float (default=100)
        number of points per 2D bin at which to begin drawing contours
    log_counts :boolean (optional)
        if True, contour levels are the base-10 logarithm of bin counts.
    histogram2d_args : dict
        keyword arguments passed to numpy.histogram2d
        see doc string of numpy.histogram2d for more information
    plot_args : dict
        keyword arguments passed to plt.plot.  By default it will use
        dict(marker='.', linestyle='none').
        see doc string of pylab.plot for more information
    contour_args : dict
        keyword arguments passed to plt.contourf or plt.contour
        see doc string of pylab.contourf for more information
    filled_contour : bool
        If True (default) use filled contours. Otherwise, use contour outlines.
    xerr, yerr : arrays pr values (optional)
        errors in x and and y dimensions. shape(N,) or shape(2, N). From
        matplotlib documentation 
        (https://matplotlib.org/api/_as_gen/matplotlib.pyplot.errorbar.html):

        "The errorbar sizes:

            scalar: Symmetric +/- values for all data points.
            shape(N,): Symmetric +/-values for each data point.
            shape(2, N): Separate - and + values for each bar. z
                First row contains the lower errors, the second 
                row contains the upper errors.
            None: No errorbar.
            Note that all error arrays should have positive values."

    ax : pylab.Axes instance
        the axes on which to plot.  If not specified, the current
        axes will be used

    Returns
    -------
    points, contours :
       points is the return value of ax.plot()
       contours is the return value of ax.contour or ax.contourf
    """

    def coerce_error_array(arr):
        """Ensures errorbar arrays are of the correct shape

        Parameters
        ----------
        
        arr : array or value
            Errorbar object to be coerced into a form that 
            can be passed to the hstack call.
        
        Returns
        -------
        coerced_arr : array
            coerced array
        """
        if arr is None:  # if no errorbars are provided
            coerced_arr = np.zeros((2, len(x)))

        elif not np.shape(arr):   # if a scalar value has been provided
            coerced_arr = arr * np.ones((2, len(x)))

        elif len(np.shape(arr)) == 1:
            coerced_arr = np.array([arr, arr])

        elif np.shape(arr)[0] > 2 and len(np.shape(arr)) > 1:
            raise ShapeError('Check shape of errorbars')

        else:
            coerced_arr = arr

        return coerced_arr
        
    x = np.asarray(x)
    y = np.asarray(y)

    default_contour_args = dict(zorder=2)
    default_plot_args = dict(marker='.', linestyle='none', zorder=1,
                             capsize=0)

    if plot_args is not None:
        default_plot_args.update(plot_args)
    plot_args = default_plot_args

    if contour_args is not None:
        default_contour_args.update(contour_args)
    contour_args = default_contour_args

    if histogram2d_args is None:
        histogram2d_args = {}

    if contour_args is None:
        contour_args = {}

    if ax is None:
        # Import here so that testing with Agg will work
        from matplotlib import pyplot as plt
        ax = plt.gca()

    H, xbins, ybins = np.histogram2d(x, y, **histogram2d_args)

    if log_counts:
        H = np.log10(1 + H)
        threshold = np.log10(1 + threshold)

    levels = np.asarray(levels)

    if levels.size == 1:
        levels = np.linspace(threshold, H.max(), levels)

    extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]

    i_min = np.argmin(levels)

    # draw a zero-width line: this gives us the outer polygon to
    # reduce the number of points we draw
    # somewhat hackish... we could probably get the same info from
    # the full contour plot below.
    outline = ax.contour(H.T, levels[i_min:i_min + 1],
                         linewidths=0, extent=extent,
                         alpha=0)

    if filled_contour:
        contours = ax.contourf(H.T, levels, extent=extent, **contour_args)
    else:
        contours = ax.contour(H.T, levels, extent=extent, **contour_args)


    xerr, yerr = coerce_error_array(xerr), coerce_error_array(yerr)

    X = np.hstack([x[:, None], y[:, None], xerr[0][:, None], 
                   xerr[1][:, None], yerr[0][:, None],
                  yerr[1][:, None]])

    if len(outline.allsegs[0]) > 0:
        outer_poly = outline.allsegs[0][0]
        try:
            # this works in newer matplotlib versions
            from matplotlib.path import Path
            points_inside = Path(outer_poly).contains_points(X[:, :2])
        except:
            # this works in older matplotlib versions
            import matplotlib.nxutils as nx
            points_inside = nx.points_inside_poly(X, outer_poly)

        Xplot = X[~points_inside]
    else:
        Xplot = X
    
    
    points = ax.errorbar(Xplot[:, 0], Xplot[:, 1], 
                         xerr=[Xplot[:, 2], Xplot[:, 3]], 
                         yerr=[Xplot[:, 4], Xplot[:, 5]], 
                         **plot_args)

    return points, contours
