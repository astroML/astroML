"""
SDSS DR7 Quasars
----------------
This example shows how to fetch the SDSS quasar photometric data, and to
plot the relationship between redshift and color.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
from matplotlib import pyplot as plt
from astroML.datasets import fetch_dr7_quasar

#------------------------------------------------------------
# Fetch the quasar data
data = fetch_dr7_quasar()

# select the first 10000 points
data = data[:10000]

r = data['mag_r']
i = data['mag_i']
z = data['redshift']

#------------------------------------------------------------
# Plot the quasar data
ax = plt.axes()
ax.plot(z, r - i, marker='.', markersize=4, linestyle='none', color='black')

ax.set_xlim(0, 5)
ax.set_ylim(-0.5, 1.0)

ax.set_xlabel('redshift')
ax.set_ylabel('r-i color')

plt.show()
