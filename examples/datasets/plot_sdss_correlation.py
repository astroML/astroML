"""
=====================================
Correlation of SDSS Star & QSO Colors
=====================================

Plot a diagram of correlations between SDSS and QSO colors
"""
import os, sys

import numpy as np
import pylab as pl

from astroML.plotting import multidensity
from astroML.datasets import fetch_sdss_colors_train

X, y = fetch_sdss_colors_train()

flag = (y == 1)
stars = X[flag]
qsos = X[~flag]
labels = ['u-g', 'g-r', 'r-i', 'i-z']

bins = [np.linspace(-0.5, 3.5, 100),
        np.linspace(0, 2, 100),
        np.linspace(-0.2, 1.8, 100),
        np.linspace(-0.2, 1.0, 100)]
multidensity(stars, labels, bins=bins)
pl.suptitle('Stars', fontsize=18)

bins = [np.linspace(-0.5, 1, 100),
        np.linspace(-0.5, 1, 100),
        np.linspace(-0.5, 1, 100),
        np.linspace(-0.5, 1, 100)]
multidensity(qsos, labels, bins=bins)
pl.suptitle('QSOs', fontsize=18)

pl.show()
