from matplotlib import pyplot as plt
import numpy as np

from .likelihood import convert_to_stdev


def plot_mcmc_contours(x, y, ax=None,
                       levels=[0.683, 0.955],
                       bins=20, **contour_args):
    if ax is None:
        ax = plt.gca()

    H, xbins, ybins = np.histogram2d(x, y, bins=bins)

    H[H == 0] = 1E-16
    Nsigma = convert_to_stdev(np.log(H))

    ax.contour(0.5 * (xbins[1:] + xbins[:-1]),
               0.5 * (ybins[1:] + ybins[:-1]),
               Nsigma.T, levels=levels, **contour_args)


def plot_mcmc(traces, labels=None, limits=None, true_values=None,
              max_posterior=None, fig=None, contour=True, scatter=False,
              levels=[0.683, 0.955], bins=20,
              bounds=[0.08, 0.08, 0.95, 0.95], **kwargs):
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
                ax.plot(limits[i], [true_values[j], true_values[j]], ':k', lw=1)
                ax.plot([true_values[i], true_values[i]], limits[j], ':k', lw=1)

            ax.set_xlim(limits[i])
            ax.set_ylim(limits[j])

            axes_list.append(ax)

    return axes_list
