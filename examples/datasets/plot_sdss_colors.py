"""
===================================
Histogram of SDSS Star & QSO Colors
===================================

Plot a histogram of the SDSS training & test colors, with gaussian fits
"""
import os, sys

import numpy as np
import pylab as pl

from astroML.pdf import GaussianProbability
from astroML.plotting import hist_with_fit
from astroML.datasets import fetch_sdss_colors_train, fetch_sdss_colors_test


def histogram_sdss_data(X, y):
    labels = ['u-g', 'g-r', 'r-i', 'i-z', 'redshift']

    pl.figure(figsize=(8, 10))

    bins = np.linspace(-4, 4, 1000)

    flag_star = (y == 1)

    for i in range(4):
        ax = pl.subplot(4, 1, 1+i)
        pl.text(0.04, 0.9, labels[i], fontsize=20,
                transform=ax.transAxes, ha='left', va='top')

        stars = X[:, i][flag_star]
        qsos = X[:, i][~flag_star]

        # fit gaussians to the data
        P_star = GaussianProbability(stars)
        P_qso = GaussianProbability(qsos)

        # plot the distribution of stars
        hist_with_fit(stars, bins, P_star(bins), bins=bins, label='stars')

        # plot the distribution of qsos
        hist_with_fit(qsos, bins, P_qso(bins), bins=bins, label='QSOs')

        pl.axis('tight')

        if i==0:
            pl.legend()

        pl.ylabel('normalized N')

        if i==3:
            pl.xlabel('color')
    

histogram_sdss_data(*fetch_sdss_colors_train())
pl.suptitle('Training Data Colors', fontsize=18)

histogram_sdss_data(*fetch_sdss_colors_test())
pl.suptitle('Test Data Colors', fontsize=18)

pl.show()
