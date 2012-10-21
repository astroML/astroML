"""
SDSS Stripe 82 Standard Stars
-----------------------------
This example shows how to use a combination scatter and contour plot to
more effectively show very dense scatter plots.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
from matplotlib import pyplot as plt

from astroML.plotting import scatter_contour
from astroML.datasets import fetch_sdss_S82standards

#------------------------------------------------------------
# Fetch the Stripe 82 standard star catalog

data = fetch_sdss_S82standards()

g = data['mmu_g']
r = data['mmu_r']
i = data['mmu_i']

#------------------------------------------------------------
# plot the results
ax = plt.axes()
scatter_contour(g - r, r - i, threshold=200, log_counts=True, ax=ax,
                histogram2d_args=dict(bins=40),
                plot_args=dict(marker='.', linestyle='none',
                               markersize=1, color='black'),
                contour_args=dict(cmap=plt.cm.bone))

ax.set_xlabel('g - r')
ax.set_ylabel('r - i')

ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)

plt.show()
