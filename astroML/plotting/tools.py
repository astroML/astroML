import os

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

from matplotlib import image
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox
from matplotlib.patches import Ellipse
import tempfile

def devectorize_axes(ax=None, dpi=None, transparent=True):
    """Convert axes contents to a png.

    This is useful when plotting many points, as the size of the saved file
    can become very large otherwise.

    Parameters
    ----------
        ax: axes instance, if None uses plt.gca()
        dpi: resolution of the png image
        transparent: make the image background transparent
    """
    if ax is None:
        ax = plt.gca()

    fig = ax.figure


    # find size of axis
    extents = ax.bbox.extents / fig.dpi
    axlim = ax.axis()

    _sp = {}
    for k in ax.spines:
        _sp[k] = ax.spines[k].get_visible()
        ax.spines[k].set_visible(False)
    _xax = ax.xaxis.get_visible()
    _yax = ax.yaxis.get_visible()
    _patch = ax.axesPatch.get_visible()
    ax.axesPatch.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # save png covering axis
    id, tmpf = tempfile.mkstemp(suffix='.png')
    plt.savefig( tmpf, format='png', dpi=dpi, transparent=transparent, bbox_inches=Bbox([extents[:2], extents[2:]]) )
    im = image.imread(tmpf)
    os.remove(tmpf)

    # clear everything on axis (but not text)
    ax.lines = []
    ax.patches = []
    ax.tables = []
    ax.artists = []
    ax.images = []
    ax.collections = []

    # show the image
    ax.imshow(im, extent=axlim, aspect='auto')

    for k in ax.spines:
        ax.spines[k].set_visible(_sp[k])
    ax.axesPatch.set_visible(_patch)
    ax.xaxis.set_visible(_xax)
    ax.yaxis.set_visible(_yax)

    if plt.isinteractive():
        plt.draw()


def discretize_cmap(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

    Parameters
    ----------
        cmap: colormap instance, eg. cm.jet.
        N: Number of colors.

    Returns
    -------
        cmap_d: discretized colormap

    Example
    -------
        >>> x = resize(arange(100), (5,100))
        >>> djet = cmap_discretize(cm.jet, 5)
    """

    cdict = cmap._segmentdata.copy()
    # N colors
    colors_i = np.linspace(0, 1., N)
    # N+1 indices
    indices = np.linspace(0, 1., N + 1)
    for key in ('red', 'green', 'blue'):
        # Find the N colors
        D = np.array(cdict[key])
        I = interpolate.interp1d(D[:, 0], D[:, 1])
        colors = I(colors_i)
        # Place these colors at the correct indices.
        A = np.zeros((N + 1, 3), float)
        A[:, 0] = indices
        A[1:, 1] = colors
        A[:-1, 2] = colors
        # Create a tuple for the dictionary.
        L = []
        for l in A:
            L.append(tuple(l))
        cdict[key] = tuple(L)
    # Return colormap object.
    return LinearSegmentedColormap('colormap', cdict, 1024)


def draw_ellipse(mu, C, scales=[1, 2, 3], ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    # find principal components and rotation angle of ellipse
    sigma_x2 = C[0, 0]
    sigma_y2 = C[1, 1]
    sigma_xy = C[0, 1]

    alpha = 0.5 * np.arctan2(2 * sigma_xy,
                             (sigma_x2 - sigma_y2))
    tmp1 = 0.5 * (sigma_x2 + sigma_y2)
    tmp2 = np.sqrt(0.25 * (sigma_x2 - sigma_y2) ** 2 + sigma_xy ** 2)

    sigma1 = np.sqrt(tmp1 + tmp2)
    sigma2 = np.sqrt(tmp1 - tmp2)

    for scale in scales:
        ax.add_patch(Ellipse((mu[0], mu[1]),
                             2 * scale * sigma1, 2 * scale * sigma2,
                             alpha * 180. / np.pi,
                             **kwargs))
