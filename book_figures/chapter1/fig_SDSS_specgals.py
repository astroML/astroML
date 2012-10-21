"""
SDSS Spectroscopic Galaxy Sample
--------------------------------
This example shows how to fetch photometric data from the SDSS spectroscopic
sample and plot a simple color-magnitude diagram.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt
from astroML.datasets import fetch_sdss_specgals

#------------------------------------------------------------
# Fetch spectroscopic galaxy data
data = fetch_sdss_specgals()
data = data[:10000]

u = data['modelMag_u']
r = data['modelMag_r']
rPetro = data['petroMag_r']

#------------------------------------------------------------
# Plot the galaxy colors and magnitudes
ax = plt.axes()
ax.plot(u - r, rPetro, '.k', markersize=4)

ax.set_xlim(1, 4.5)
ax.set_ylim(18, 13.5)

ax.set_xlabel(r'$\mathrm{u - r}$')
ax.set_ylabel(r'$\mathrm{r_{petrosian}}$')

plt.show()
