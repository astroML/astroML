"""
Multi-panel plotting
"""
from copy import deepcopy
import numpy as np


class MultiAxes(object):
    """Visualize Multiple-dimensional data

    This class enables the visualization of multi-dimensional data, using
    a triangular grid of 2D plots.

    Parameters
    ----------
    ndim : integer
        Number of data dimensions
    inner_labels : bool
        If true, then label the inner axes.  If false, then only the outer
        axes will be labeled
    fig : matplotlib.Figure
        if specified, draw the plot on this figure.  Otherwise, use the
        current active figure.
    left, bottom, right, top, wspace, hspace : floats
        these parameters control the layout of the plots.  They behave have
        an identical effect as the arguments to plt.subplots_adjust.  If not
        specified, default values from the rc file will be used.

    Examples
    --------
    A grid of scatter plots can be created as follows::

        x = np.random.normal((4, 1000))
        R = np.random.random((4, 4))  # projection matrix
        x = np.dot(R, x)
        ax = MultiAxes(4)
        ax.scatter(x)
        ax.set_labels(['x1', 'x2', 'x3', 'x4'])

    Alternatively, the scatter plot can be visualized as a density::

        ax = MultiAxes(4)
        ax.density(x, bins=[20, 20, 20, 20])
    """
    def __init__(self, ndim, inner_labels=False,
                 fig=None,
                 left=None, bottom=None,
                 right=None, top=None,
                 wspace=None, hspace=None):
        # Import here so that testing with Agg will work
        from matplotlib import pyplot as plt
        if fig is None:
            fig = plt.gcf()
        self.fig = fig

        self.ndim = ndim
        self.inner_labels = inner_labels

        self._update('left', left)
        self._update('bottom', bottom)
        self._update('right', right)
        self._update('top', top)
        self._update('wspace', wspace)
        self._update('hspace', hspace)

        self.axes = self._draw_panels()

    def _update(self, s, val):
        # Import here so that testing with Agg will work
        from matplotlib import rcParams
        if val is None:
            val = getattr(self, s, None)
            if val is None:
                key = 'figure.subplot.' + s
                val = rcParams[key]
        setattr(self, s, val)

    def _check_data(self, data):
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("data dimension should be 2")
        if data.shape[1] != self.ndim:
            raise ValueError("leading dimension of data should match ndim")
        return data

    def _draw_panels(self):
        # Import here so that testing with Agg will work
        from matplotlib import pyplot as plt
        if self.top <= self.bottom:
            raise ValueError('top must be larger than bottom')
        if self.right <= self.left:
            raise ValueError('right must be larger than left')

        ndim = self.ndim

        panel_width = ((self.right - self.left)
                       / (ndim - 1 + self.wspace * (ndim - 2)))
        panel_height = ((self.top - self.bottom)
                        / (ndim - 1 + self.hspace * (ndim - 2)))

        full_panel_width = (1 + self.wspace) * panel_width
        full_panel_height = (1 + self.hspace) * panel_height

        axes = np.empty((ndim, ndim), dtype=object)
        axes.fill(None)

        for j in range(1, ndim):
            for i in range(j):
                left = self.left + i * full_panel_width
                right = self.bottom + (ndim - 1 - j) * full_panel_height
                ax = self.fig.add_axes([left, right,
                                        panel_width, panel_height])
                axes[i, j] = ax

        if not self.inner_labels:
            # remove unneeded x labels
            for i in range(ndim):
                for j in range(ndim - 1):
                    ax = axes[i, j]
                    if ax is not None:
                        ax.xaxis.set_major_formatter(plt.NullFormatter())

            # remove unneeded y labels
            for i in range(1, ndim):
                for j in range(ndim):
                    ax = axes[i, j]
                    if ax is not None:
                        ax.yaxis.set_major_formatter(plt.NullFormatter())

        return np.asarray(axes, dtype=object)

    def set_limits(self, limits):
        """Set the axes limits

        Parameters
        ----------
        limits : list of tuples
            a list of plot limits for each dimension, each in the form
            (xmin, xmax).  The length of `limits` should match the data
            dimension.
        """
        if len(limits) != self.ndim:
            raise ValueError("limits do not match number of dimensions")

        for i in range(self.ndim):
            for j in range(self.ndim):
                ax = self.axes[i, j]
                if ax is not None:
                    ax.set_xlim(limits[i])
                    ax.set_ylim(limits[j])

    def set_labels(self, labels):
        """Set the axes labels

        Parameters
        ----------
        labels : list of strings
            a list of plot limits for each dimension.  The length of `labels`
            should match the data dimension.
        """
        if len(labels) != self.ndim:
            raise ValueError("labels do not match number of dimensions")

        for i in range(self.ndim):
            ax = self.axes[i, self.ndim - 1]
            if ax is not None:
                ax.set_xlabel(labels[i])

        for j in range(self.ndim):
            ax = self.axes[0, j]
            if ax is not None:
                ax.set_ylabel(labels[j])

    def set_locators(self, locators):
        """Set the tick locators for the plots

        Parameters
        ----------
        locators : list or plt.Locator object
            If a list, then the length should match the data dimension.  If
            a single Locator instance, then each axes will be given the
            same locator.
        """
        # Import here so that testing with Agg will work
        from matplotlib import pyplot as plt
        if isinstance(locators, plt.Locator):
            locators = [deepcopy(locators) for i in range(self.ndim)]
        elif len(locators) != self.ndim:
            raise ValueError("locators do not match number of dimensions")

        for i in range(self.ndim):
            for j in range(self.ndim):
                ax = self.axes[i, j]
                if ax is not None:
                    ax.xaxis.set_major_locator(locators[i])
                    ax.yaxis.set_major_locator(locators[j])

    def set_formatters(self, formatters):
        """Set the tick formatters for the outer edge of plots

        Parameters
        ----------
        formatterss : list or plt.Formatter object
            If a list, then the length should match the data dimension.  If
            a single Formatter instance, then each axes will be given the
            same locator.
        """
        # Import here so that testing with Agg will work
        from matplotlib import pyplot as plt
        if isinstance(formatters, plt.Formatter):
            formatters = [deepcopy(formatters) for i in range(self.ndim)]
        elif len(formatters) != self.ndim:
            raise ValueError("formatters do not match number of dimensions")

        for i in range(self.ndim):
            ax = self.axes[i, self.ndim - 1]
            if ax is not None:
                ax.xaxis.set_major_formatter(formatters[i])

        for j in range(self.ndim):
            ax = self.axes[0, j]
            if ax is not None:
                ax.xaxis.set_major_formatter(formatters[i])

    def plot(self, data, *args, **kwargs):
        """Plot data

        This function calls plt.plot() on each axes.  All arguments or
        keyword arguments are passed to the plt.plot function.

        Parameters
        ----------
        data : ndarray
            shape of data is [n_samples, ndim], and ndim should match that
            passed to the MultiAxes constructor.
        """
        data = self._check_data(data)

        for i in range(self.ndim):
            for j in range(self.ndim):
                ax = self.axes[i, j]
                if ax is None:
                    continue
                ax.plot(data[:, i], data[:, j], *args, **kwargs)

    def scatter(self, data, *args, **kwargs):
        """Scatter plot data

        This function calls plt.scatter() on each axes.  All arguments or
        keyword arguments are passed to the plt.scatter function.

        Parameters
        ----------
        data : ndarray
            shape of data is [n_samples, ndim], and ndim should match that
            passed to the MultiAxes constructor.
        """
        data = self._check_data(data)

        for i in range(self.ndim):
            for j in range(self.ndim):
                ax = self.axes[i, j]
                if ax is None:
                    continue
                ax.scatter(data[:, i], data[:, j], *args, **kwargs)

    def density(self, data, bins=20, **kwargs):
        """Density plot of data

        This function calls np.histogram2D to bin the data in each axes, then
        calls plt.imshow() on the result.  All extra arguments or
        keyword arguments are passed to the plt.imshow function.

        Parameters
        ----------
        data : ndarray
            shape of data is [n_samples, ndim], and ndim should match that
            passed to the MultiAxes constructor.
        bins : int, array, list of ints, or list of arrays
            specify the bins for each dimension. If bins is a list, then the
            length must match the data dimension
        """
        data = self._check_data(data)

        if not hasattr(bins, '__len__'):
            bins = [bins for i in range(self.ndim)]
        elif len(bins) != self.ndim:
            bins = [bins for i in range(self.ndim)]

        for i in range(self.ndim):
            for j in range(self.ndim):
                ax = self.axes[i, j]
                if ax is None:
                    continue

                H, xbins, ybins = np.histogram2d(data[:, i], data[:, j],
                                                 (bins[i], bins[j]))
                ax.imshow(H.T, origin='lower', aspect='auto',
                          extent=(xbins[0], xbins[-1], ybins[0], ybins[-1]),
                          **kwargs)

                ax.set_xlim(xbins[0], xbins[-1])
                ax.set_ylim(ybins[0], ybins[-1])
