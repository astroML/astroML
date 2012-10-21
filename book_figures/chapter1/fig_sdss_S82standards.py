"""
SDSS Stripe 82 Standard Stars
-----------------------------
This example shows how to plot a color-color diagram for the SDSS Stripe 82
standard stars
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
from matplotlib import pyplot as plt
from astroML.datasets import fetch_sdss_S82standards

#------------------------------------------------------------
# Fetch the stripe 82 data
data = fetch_sdss_S82standards()

# select the first 10000 points
data = data[:10000]

# select the mean magnitudes for g, r, i
g = data['mmu_g']
r = data['mmu_r']
i = data['mmu_i']

#------------------------------------------------------------
# Plot the g-r vs r-i colors
ax = plt.axes()
ax.plot(g - r, r - i, marker='.', markersize=2,
        color='black', linestyle='none')

ax.set_xlabel('g - r')
ax.set_ylabel('r - i')

plt.show()
