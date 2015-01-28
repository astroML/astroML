import numpy as np


def convert_to_stdev(logL):
    """
    Given a grid of log-likelihood values, convert them to cumulative
    standard deviation.  This is useful for drawing contours from a
    grid of likelihoods.
    """
    sigma = np.exp(logL)

    shape = sigma.shape
    sigma = sigma.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(sigma)[::-1]
    i_unsort = np.argsort(i_sort)

    sigma_cumsum = sigma[i_sort].cumsum()
    sigma_cumsum /= sigma_cumsum[-1]

    return sigma_cumsum[i_unsort].reshape(shape)


def plot_mcmc(traces, labels=None, limits=None, true_values=None,
              fig=None, contour=True, scatter=False,
              levels=[0.683, 0.955], bins=20,
              bounds=[0.08, 0.08, 0.95, 0.95], **kwargs):
    """Plot a grid of MCMC results

    Parameters
    ----------
    traces : array_like
        the MCMC chain traces.  shape is [Ndim, Nchain]
    labels : list of strings (optional)
        if specified, the label associated with each trace
    limits : list of tuples (optional)
        if specified, the axes limits for each trace
    true_values : list of floats (optional)
        if specified, the true value for each trace (will be indicated with
        an 'X' on the plot)
    fig : matplotlib.Figure (optional)
        the figure on which to draw the axes.  If not specified, a new one
        will be created.
    contour : bool (optional)
        if True, then draw contours in each subplot.  Default=True.
    scatter : bool (optional)
        if True, then scatter points in each subplot.  Default=False.
    levels : list of floats
        the list of percentile levels at which to plot contours.  Each
        entry should be between 0 and 1
    bins : int, tuple, array, or tuple of arrays
        the binning parameter passed to np.histogram2d.  It is assumed that
        the point density is constant on the scale of the bins
    bounds : list of floats
        the bounds of the set of axes used for plotting

    additional keyword arguments are passed to scatter() and contour()

    Returns
    -------
    axes_list : list of matplotlib.Axes instances
        the list of axes created by the routine
    """
    # Import here so that testing with Agg will work
    from matplotlib import pyplot as plt

    if fig is None:
        fig = plt.figure(figsize=(8, 8))

    if limits is None:
        limits = [(t.min(), t.max()) for t in traces]

    if labels is None:
        labels = ['' for t in traces]

    num_traces = len(traces)

    bins = [np.linspace(limits[i][0], limits[i][1], bins + 1)
            for i in range(num_traces)]

    xmin, xmax = bounds[0], bounds[2]
    ymin, ymax = bounds[1], bounds[3]

    dx = (xmax - xmin) * 1. / (num_traces - 1)
    dy = (ymax - ymin) * 1. / (num_traces - 1)

    axes_list = []

    for j in range(1, num_traces):
        for i in range(j):
            ax = fig.add_axes([xmin + i * dx,
                               ymin + (num_traces - 1 - j) * dy,
                               dx, dy])

            if scatter:
                plt.scatter(traces[i], traces[j], **kwargs)

            if contour:
                H, xbins, ybins = np.histogram2d(traces[i], traces[j],
                                                 bins=(bins[i], bins[j]))

                H[H == 0] = 1E-16
                Nsigma = convert_to_stdev(np.log(H))

                ax.contour(0.5 * (xbins[1:] + xbins[:-1]),
                           0.5 * (ybins[1:] + ybins[:-1]),
                           Nsigma.T, levels=levels, **kwargs)

            if i == 0:
                ax.set_ylabel(labels[j])
            else:
                ax.yaxis.set_major_formatter(plt.NullFormatter())

            if j == num_traces - 1:
                ax.set_xlabel(labels[i])
            else:
                ax.xaxis.set_major_formatter(plt.NullFormatter())

            if true_values is not None:
                ax.plot(limits[i], [true_values[j], true_values[j]],
                        ':k', lw=1)
                ax.plot([true_values[i], true_values[i]], limits[j],
                        ':k', lw=1)

            ax.set_xlim(limits[i])
            ax.set_ylim(limits[j])

            axes_list.append(ax)

    return axes_list
