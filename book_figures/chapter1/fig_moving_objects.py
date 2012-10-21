"""
SDSS Moving Object Data
-----------------------
This example shows how to fetch the moving object (i.e. asteroid) data from
Stripe 82 and to plot some measures of the orbital dynamics.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
from matplotlib import pyplot as plt
from astroML.datasets import fetch_moving_objects

#------------------------------------------------------------
# Fetch the moving object data
data = fetch_moving_objects(Parker2008_cuts=True)

# Use only the first 10000 points
data = data[:10000]

a = data['aprime']
sini = data['sin_iprime']

#------------------------------------------------------------
# Plot the results
ax = plt.axes()
ax.plot(a, sini, '.', markersize=2, color='black')

ax.set_xlabel('Semi-major Axis (AU)')
ax.set_ylabel('Sine of Inclination Angle')

plt.show()
