"""
=====================================
Correlation of SDSS Star & QSO Colors
=====================================

Plot a diagram of correlations between SDSS and QSO colors
"""
import os, sys

import numpy as np
import pylab as pl

from astroML.pdf import GaussianProbability
from astroML.plotting import multicontour
from astroML.datasets import fetch_sdss_colors_train

X, y = fetch_sdss_colors_train()

flag = (y == 1)
stars = X[flag]
qsos = X[~flag]
labels = ['u-g', 'g-r', 'r-i', 'i-z']

fig = pl.figure(figsize=(10,10))

bins = [np.linspace(-0.5, 3.5, 50),
        np.linspace(0, 2, 50),
        np.linspace(-0.2, 1.8, 50),
        np.linspace(-0.2, 1.0, 50)]
ax0 = multicontour(stars, labels, bins=bins, fig=fig, colors='b')
l1 = ax0[0,0].collections[-1]

bins = [np.linspace(-0.5, 1, 50),
        np.linspace(-0.5, 1, 50),
        np.linspace(-0.5, 1, 50),
        np.linspace(-0.5, 1, 50)]
ax1 = multicontour(qsos, labels, bins=bins, fig=fig, colors='r')
l2 = ax1[0,0].collections[-1]

pl.suptitle('QSOs and Stars', fontsize=18)
pl.figlegend([l1, l2], ['Stars','QSOs'], (0.7, 0.7))
pl.show()
