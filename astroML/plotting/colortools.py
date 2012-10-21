import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import colorConverter, LinearSegmentedColormap

def get_faded(c, fade=0.5):
    c = colorConverter.to_rgb(c)
    return tuple([ci + fade * (1. - ci) for ci in c])

class ColorWheel_gen:
    """Generator class for a color wheel."""
    def __init__(self, colorwheel='brgmcy'):
        self.colorwheel = colorwheel
        self.ax_dict = {}

    def next(self, fade=0, ax=None):
        if ax is None:
            ax = plt.gca()

        if ax in self.ax_dict:
            i = (self.ax_dict[ax] + 1) % len(self.colorwheel)
        else:
            i = self.ax_dict[ax] = 0
        
        return get_faded(self.colorwheel[i], fade)

    def current(self, fade=0, ax=None):
        if ax is None:
            ax = plt.gca()

        if ax in self.ax_dict:
            i = (self.ax_dict[ax] + 1) % len(self.colorwheel)
        else:
            i = self.ax_dict[ax] = 0
        
        return get_faded(self.colorwheel[i], fade)

        
colorWheel = ColorWheel_gen()

# 2D color tools
_cdict_RdBu = {
'red' : ((0., 1., 1.), (1., 0., 0.)),
'green': ((0., 0., 0.), (1., 0., 0.)),
'blue' : ((0., 0., 0.), (1., 1., 1.))
}

RdBu = LinearSegmentedColormap('RdBu', _cdict_RdBu)

_cdict_GnRd = {
'red' : ((0., 0., 0.), (1., 1., 1.)),
'green': ((0., 1., 1.), (1., 0., 0.)),
'blue' : ((0., 0., 0.), (1., 0., 0.))
}

GnRd = LinearSegmentedColormap('GnRd', _cdict_GnRd)

class Colormap2D:
    """
    This is a simple class which creates a 2D colormap from two
    1D colormaps. For the 1D colormaps, any standard matplotlib Colormap
    instance can be used.  The default choices give a good, bright range
    of colors.
    """
    def __init__(self, cmap1=RdBu, cmap2=GnRd):
        self.cmap1 = plt.cm.get_cmap(cmap1)
        self.cmap2 = plt.cm.get_cmap(cmap2)

    def __call__(self, X):
        """
        Parameters
        ----------
        X : array, last dimension of size 2
            Array of 2-dimensional data to colorize.
            
        Returns
        -------
        c : array, shape = X.shape[:-1] + (4,)
            Array of RGBA tuples giving the color of each point in X.
        """
        X = np.atleast_2d(X)
        assert X.shape[-1] == 2
        Xroll = np.rollaxis(X, -1)

        c = self.cmap1(Xroll[0]) + self.cmap2(Xroll[1])
        c /= c.max(0)

        return c
