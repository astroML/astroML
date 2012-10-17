"""
K-means Classification
----------------------

Perform K-means classification on the SDSS colors dataset, plotting
the result.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning for Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
import pylab as pl

from sklearn.cluster import KMeans
from astroML.plotting import multiscatter, multidensity, multicontour
from astroML.datasets import fetch_sdss_colors_train

X, y = fetch_sdss_colors_train()

km = KMeans(k=4).fit(X)

C = km.cluster_centers_

# plot the input data
colors = 'kyrc'
bins = [np.linspace(-0.5, 3.5, 100),
        np.linspace(-0.5, 2, 100),
        np.linspace(-0.5, 1.8, 100),
        np.linspace(-0.5, 1.0, 100)]

fig = pl.figure(figsize=(10, 10))

multidensity(X, ['u-g', 'g-r', 'r-i', 'i-z'], bins=bins, fig=fig)
multiscatter(C, ['u-g', 'g-r', 'r-i', 'i-z'],
             fig=fig, scatter_kwargs=dict(edgecolors='k',
                                          facecolors=colors,
                                          marker='o', s=40))

# over-plot contours showing the K-means classification
y_pred = km.predict(X)
vals = np.unique(y_pred)
bins = [np.linspace(-0.5, 3.5, 25),
        np.linspace(-0.5, 2, 25),
        np.linspace(-0.5, 1.8, 25),
        np.linspace(-0.5, 1.0, 25)]
for i, v in enumerate(vals):
    multicontour(X[y_pred == v], ['u-g', 'g-r', 'r-i', 'i-z'],
                 bins=bins, fig=fig, colors=colors[i])

pl.suptitle('K-means Classification on SDSS Photometry')
pl.show()
