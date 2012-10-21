from matplotlib import pyplot as plt
from hist import hist_with_fit, hist
from multiscatter import multiscatter, multidensity, multicontour
from multiimshow import multiimshow
from scatter_contour import scatter_contour
from mcmc import plot_mcmc

def plot_tissot_ellipse(longitude, latitude, radius, ax=None, **kwargs):
    """Plot Tissot Ellipse/Tissot Indicatrix

    Parameters
    ----------
    longitude : float or array_like
        longitude of ellipse centers (radians)
    latitude : float or array_like
        latitude of ellipse centers (radians)
    radius : float or array_like
        radius of ellipses
    ax : Axes object (optional)
        matplotlib axes instance on which to draw ellipses.

    Other Parameters
    ----------------
    other keyword arguments will be passed to matplotlib.patches.Ellipse.
    """
    import numpy as np
    from matplotlib.patches import Ellipse

    if ax is None:
        ax = plt.gca()

    for long, lat, rad in np.broadcast(longitude, latitude, radius):
        el = Ellipse((long, lat), radius/np.cos(lat), radius, **kwargs)
        ax.add_patch(el)
    
