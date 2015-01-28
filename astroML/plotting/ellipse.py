import numpy as np


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
    # Import here so that testing with Agg will work
    from matplotlib import pyplot as plt
    from matplotlib.patches import Ellipse

    if ax is None:
        ax = plt.gca()

    for long, lat, rad in np.broadcast(longitude, latitude, radius):
        el = Ellipse((long, lat), radius / np.cos(lat), radius, **kwargs)
        ax.add_patch(el)
