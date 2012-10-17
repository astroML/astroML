"""
===============
Scatter-Contour
===============

This example shows how to plot a scatter plot with contours in the locations
where the data points are dense.
"""

import numpy as np
import pylab as pl
from astroML.plotting import scatter_contour

# create a correlated & normalized dataset
X = np.random.normal(size=(2, 10000))
M = np.array([[1,1],[0,1]])
x,y = np.dot(M, X)
    
# plot using scatter_contour
scatter_contour(x, y,
                histogram2d_args=dict(bins=20),
                scatter_args=dict(s=4, lw=0, c='k'),
                contour_args=dict(cmap=pl.cm.bone))
pl.show()

