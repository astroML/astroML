import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker

def multiimshow(data, extents, labels,
                xspacing=0.01, yspacing=0.01,
                left=0.1, right=0.1, bottom=0.1, top=0.1,
                fig=None,
                imshow_kwargs={}):
    """
    show a multi-panel projection of a data cube

    Parameters
    ----------
    data : array, shape = (n_features, n_features, N, N)
        data[i,j] is the image showing the density of feature i vs feature j
    extents : array, shape = (n_features, 2)
        extents[i] gives the extent of feature i
    """
    if fig is None:
        fig = plt.figure(figsize=(10,10))

    n_features = data.ndim
    
    xsize = (1. - left - right - xspacing * (n_features - 2)) / (n_features - 1)
    ysize = (1. - top - bottom - yspacing * (n_features - 2)) / (n_features - 1)

    xlocs = np.arange(left, 1 - right, xsize + xspacing)
    ylocs = np.arange(bottom, 1 - top, ysize + yspacing)

    ax_list = np.zeros((n_features - 1, n_features - 1),
                       dtype=object)

    for i in range(n_features - 1):
        for j in range(n_features - 1 - i):
            ax = fig.add_axes([xlocs[i], ylocs[j], xsize, ysize])
            ax_list[i, j] = ax

            kwargs = imshow_kwargs.copy()
            kwargs['extent'] = (tuple(extents[i])
                                + tuple(extents[n_features - 1 - j]))
            kwargs['aspect'] = 'auto'
            kwargs['origin'] = 'lower'

            ax.imshow(data[i, n_features - 1 - j], **kwargs)
                
            if i == 0:
                ax.set_ylabel(labels[n_features - 1 - j])
            else:
                ax.yaxis.set_major_formatter(ticker.NullFormatter())

            if j == 0:
                ax.set_xlabel(labels[i])
            else:
                ax.xaxis.set_major_formatter(ticker.NullFormatter())


    return ax_list
