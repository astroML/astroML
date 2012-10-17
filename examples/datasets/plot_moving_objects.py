"""
====================================
SDSS Stripe 82 Moving Object Catalog
====================================

This plot demonstrates how to fetch data from the SDSS Moving object catalog,
and plot using a multicolor plot similar to that used in figures 3-4 of
Parker et al 2008.
"""

import numpy as np
import pylab as pl
from astroML.datasets import fetch_moving_objects

from matplotlib._cm import datad
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import register_cmap

MO_cmap_data = dict(red = ((0.00, 0.0, 0.0),
                           (0.15, 0.0, 0.0),
                           (0.35, 1.0, 1.0),
                           (0.75, 1.0, 1.0),
                           (1.00, 0.0, 0.0)),
                    blue = ((0.00, 0.7, 0.7),
                            (0.15, 1.0, 1.0),
                            (0.40, 0.0, 0.0),
                            (0.75, 0.0, 0.0),
                            (1.00, 0.0, 0.0)),
                    green = ((0.00, 0.0, 0.0),
                             (0.40, 0.0, 0.0),
                             (0.60, 0.5, 0.5),
                             (0.85, 0.7, 0.7),
                             (1.00, 0.7, 0.7)))
MO_cmap = LinearSegmentedColormap('MO_cmap', MO_cmap_data)

def black_bg_fig_ax():
    """Create a figure and axis with black background"""
    fig = pl.figure(facecolor='k')
    ax = pl.axes(axisbg='k')

    # set ticks and labels to white
    for spine in ax.spines.values():
        spine.set_color('w')
    
    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        for child in tick.get_children():
            child.set_color('w')
    
    return fig, ax


def compute_color(mag_a, mag_i, mag_z):
    """Compute colors for the plots from the magnitudes"""
    iz = mag_i - mag_z

    amin = mag_a.min()
    amax = mag_a.max()
    izmin = iz.min()
    izmax = iz.max()

    anrm = (mag_a + 0.0) / (amax - amin)
    iznrm = (iz + 0.15) / (izmax - izmin)

    color = np.arctan2(iznrm, anrm) / np.pi

    color[color < -0.5] += 2
    color -= 0.5

    return -1. / (1 + np.exp(-5 * (color + 0.25)))


data = fetch_moving_objects(Parker2008_cuts=True)
mag_a = data['mag_a']
mag_i = data['mag_i']
mag_z = data['mag_z']
a = data['aprime']
sini = data['sin_iprime']

# add some random scatter:
#  magnitudes are good only to 0.01
mag_a += 0.01 * np.random.random(size=mag_a.shape)
mag_i += 0.01 * np.random.random(size=mag_i.shape)
mag_z += 0.01 * np.random.random(size=mag_z.shape)

color = compute_color(mag_a, mag_i, mag_z)

#------------------------------------------------------------
# plot the color-magnitude plot
fig, ax = black_bg_fig_ax()
pl.scatter(mag_a, mag_i - mag_z,
           c=color, s=1, lw=0,
           cmap=MO_cmap)

pl.plot([0, 0], [-0.8, 0.6], '--w', lw=2)
pl.plot([0, 0.4], [-0.15, -0.15], '--w', lw=2)

pl.xlim(-0.3, 0.4)
pl.ylim(-0.8, 0.6)

# label the plot
pl.xlabel('a*', color='w')
pl.ylabel('i-z', color='w')

#------------------------------------------------------------
# plot the orbital parameters plot
fig, ax = black_bg_fig_ax()
pl.scatter(a, sini,
           c=color, s=1, lw=0,
           cmap=MO_cmap)

pl.plot([2.5, 2.5], [-0.02, 0.3], '--w')
pl.plot([2.82, 2.82], [-0.02, 0.3], '--w')
pl.xlim(2.0, 3.3)
pl.ylim(-0.02, 0.3)

# label the plot
text_kwargs = dict(color='w', fontsize=14,
                   transform=pl.gca().transAxes,
                   ha='center', va='bottom')

pl.text(0.25, 1.01, 'Inner', **text_kwargs)

pl.text(0.53, 1.01, 'Mid', **text_kwargs)

pl.text(0.83, 1.01, 'Outer', **text_kwargs)

pl.xlabel('a (AU)', color='w')
pl.ylabel('sin(i)', color='w')

pl.show()
