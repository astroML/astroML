import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker


def densityplot(x, y, bins=None, cmap=plt.cm.jet, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    H, xbins, ybins = np.histogram2d(x, y, bins)
    ax.imshow(H.T, origin='lower',
              extent=(xbins[0], xbins[-1], ybins[0], ybins[-1]),
              aspect='auto', **kwargs)


def multidensity(data, labels=None, bins=None,
                 xspacing=0.01, yspacing=0.01,
                 left=0.1, right=0.1, bottom=0.1, top=0.1,
                 fig=None,
                 kwargs={}):
    """
    Make a multiple-panel scatter-plot
    """
    if fig is None:
        fig = plt.figure(figsize=(10, 10))

    n_samples, n_features = data.shape

    if bins is None:
        bins = n_features * [100]

    xsize = ((1. - left - right - xspacing * (n_features - 2))
             / (n_features - 1))
    ysize = ((1. - top - bottom - yspacing * (n_features - 2))
             / (n_features - 1))

    xlocs = np.arange(left, 1 - right, xsize + xspacing)
    ylocs = np.arange(bottom, 1 - top, ysize + yspacing)

    ax_list = np.zeros((n_features - 1, n_features - 1),
                       dtype=object)

    for i in range(n_features - 1):
        for j in range(n_features - 1 - i):
            ax = fig.add_axes([xlocs[i], ylocs[j], xsize, ysize])
            ax_list[i, j] = ax

            densityplot(data[:, i], data[:, n_features - 1 - j],
                        (bins[i], bins[n_features - 1 - j]),
                        **kwargs)

            if i == 0:
                ax.set_ylabel(labels[n_features - 1 - j])
            else:
                ax.yaxis.set_major_formatter(ticker.NullFormatter())

            if j == 0:
                ax.set_xlabel(labels[i])
                ticklabels = ax.get_xticklabels()
                for label in ticklabels:
                    label.set_rotation(90)
            else:
                ax.xaxis.set_major_formatter(ticker.NullFormatter())

    return ax_list
