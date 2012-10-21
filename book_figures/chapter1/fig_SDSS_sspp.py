"""
SDSS Segue Stellar Parameter Pipeline Data
------------------------------------------
This example shows how to fetch the data from the Segue Stellar Parameter
Pipeline (SSPP), and plot the temperature and surface gravity of a selection
of stars.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
from matplotlib import pyplot as plt
from astroML.datasets import fetch_sdss_sspp

#------------------------------------------------------------
# Fetch the data
data = fetch_sdss_sspp()

# select the first 10000 points
data = data[:10000]

# do some reasonable magnitude cuts
rpsf = data['rpsf']
data = data[(rpsf > 15) & (rpsf < 19)]

# get the desired data
logg = data['logg']
Teff = data['Teff']

#------------------------------------------------------------
# Plot the data
ax = plt.axes()
ax.plot(Teff, logg, marker='.', markersize=2, linestyle='none', color='black')

ax.set_xlim(8000, 4500)
ax.set_ylim(5, 1)

ax.set_xlabel(r'$\mathrm{T_{eff}\ (K)}$')
ax.set_ylabel(r'$\mathrm{log_{10}[g / (cm/s^2)]}$')

plt.show()
