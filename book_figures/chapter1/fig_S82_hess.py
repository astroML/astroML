"""
SDSS Stripe 82 Hess Diagram
---------------------------
This example shows how to create a hess diagram of the SDSS Stripe 82 data,
using the 2D histogram function in numpy.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt

from astroML.datasets import fetch_sdss_S82standards

#------------------------------------------------------------
# Fetch the stripe 82 data
data = fetch_sdss_S82standards()

g = data['mmu_g']
r = data['mmu_r']
i = data['mmu_i']

#------------------------------------------------------------
# Compute and plot the 2D histogram
H, xbins, ybins = np.histogram2d(g - r, r - i,
                                 bins=(np.linspace(-0.5, 2.5, 50),
                                       np.linspace(-0.5, 2.5, 50)))

# Create a black and white color map where bad data (NaNs) are white
cmap = plt.cm.binary
cmap.set_bad('w', 1.)

# Use the image display function imshow() to plot the result
ax = plt.axes()
ax.imshow(np.log10(H).T, origin='lower',
          extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
          cmap=cmap, interpolation='nearest',
          aspect='auto')

ax.set_xlabel('g - r')
ax.set_ylabel('r - i')

ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)

plt.show()
