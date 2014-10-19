import os

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

from matplotlib import image
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox
from matplotlib.patches import Ellipse
from ..py3k_compat import BytesIO


def devectorize_axes(ax=None, dpi=None, transparent=True):
    """Convert axes contents to a png.

    This is useful when plotting many points, as the size of the saved file
    can become very large otherwise.

    Parameters
    ----------
    ax : Axes instance (optional)
        Axes to de-vectorize.  If None, this uses the current active axes
        (plt.gca())
    dpi: int (optional)
        resolution of the png image.  If not specified, the default from
        'savefig.dpi' in rcParams will be used
    transparent : bool (optional)
        if True (default) then the PNG will be made transparent

    Returns
    -------
    ax : Axes instance
        the in-place modified Axes instance

    Examples
    --------
    The code can be used in the following way::

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        x, y = np.random.random((2, 10000))
        ax.scatter(x, y)
        devectorize_axes(ax)
        plt.savefig('devectorized.pdf')

    The resulting figure will be much smaller than the vectorized version.
    """
    if ax is None:
        ax = plt.gca()

    fig = ax.figure
    axlim = ax.axis()

    # setup: make all visible spines (axes & ticks) & text invisible
    # we need to set these back later, so we save their current state
    _sp = {}
    _txt_vis = [t.get_visible() for t in ax.texts]
    for k in ax.spines:
        _sp[k] = ax.spines[k].get_visible()
        ax.spines[k].set_visible(False)
    for t in ax.texts:
        t.set_visible(False)

    _xax = ax.xaxis.get_visible()
    _yax = ax.yaxis.get_visible()
    _patch = ax.axesPatch.get_visible()
    ax.axesPatch.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # convert canvas to PNG
    extents = ax.bbox.extents / fig.dpi
    output = BytesIO()
    plt.savefig(output, format='png', dpi=dpi,
                transparent=transparent,
                bbox_inches=Bbox([extents[:2], extents[2:]]))
    output.seek(0)
    im = image.imread(output)

    # clear everything on axis (but not text)
    ax.lines = []
    ax.patches = []
    ax.tables = []
    ax.artists = []
    ax.images = []
    ax.collections = []

    # Show the image
    ax.imshow(im, extent=axlim, aspect='auto', interpolation='nearest')

    # restore all the spines & text
    for k in ax.spines:
        ax.spines[k].set_visible(_sp[k])
    for t, v in zip(ax.texts, _txt_vis):
        t.set_visible(v)
    ax.axesPatch.set_visible(_patch)
    ax.xaxis.set_visible(_xax)
    ax.yaxis.set_visible(_yax)

    if plt.isinteractive():
        plt.draw()

    return ax


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
