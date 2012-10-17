"""
=============
2D Color Plot
=============

A simple example showing how to scatter-plot points with a 2-dimensional
color map.  This also shows how to create and save a plot with a black
background.
"""

import numpy as np
import pylab as pl
import matplotlib as mpl
from astroML.plotting.colortools import Colormap2D

# create random data
X = np.random.random((2000, 2))

# create a figure and axis with black background
fig = pl.figure(facecolor='k')
ax = pl.axes(axisbg='k')

# set ticks and labels to white
for spine in ax.spines.values():
    spine.set_color('w')
    
for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
    for child in tick.get_children():
        child.set_color('w')

# scatter-plot
cmap = Colormap2D()
pl.scatter(X[:,0], X[:,1], c=cmap(X), s=10, lw=0)

pl.title('Title', color='w')
pl.xlabel('x label', color='w')
pl.ylabel('y label', color='w')

# to save figure with the black background, use the following command:
#
#fig.savefig('figure.png',
#            facecolor=fig.get_facecolor(), edgecolor='none')

pl.show()
