"""
Stellar Parameters Hess Diagram
-------------------------------
This example shows how to create Hess diagrams of the Segue Stellar Parameters
Pipeline (SSPP) data to show multiple features on a single plot.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning for Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
import pylab as pl

#------------------------------------------------------------
# Get SDSS SSPP data
from astroML.datasets import fetch_sdss_sspp
data = fetch_sdss_sspp()

# do some reasonable magnitude cuts
rpsf = data['rpsf']
data = data[(rpsf > 15) & (rpsf < 19)]

# get the desired data
logg = data['logg']
Teff = data['Teff']
FeH = data['FeH']

#------------------------------------------------------------
# Plot the results using the binned_statistic function
from astroML.stats import binned_statistic_2d
N, xedges, yedges = binned_statistic_2d(Teff, logg, FeH,
                                        'count', bins=100)
FeH_mean, xedges, yedges = binned_statistic_2d(Teff, logg, FeH,
                                               'mean', bins=100)

# Define custom colormaps: Set pixels with no sources to white
cmap = pl.cm.copper
cmap.set_bad('w', 1.)

cmap_multicolor = pl.cm.jet
cmap_multicolor.set_bad('w', 1.)

# Create figure and subplots
fig = pl.figure(figsize=(10, 4))
fig.subplots_adjust(wspace=0.22, left=0.08, right=0.95,
                    bottom=0.07, top=0.95)

#--------------------
# First axes:
pl.subplot(131, xticks=[4000, 5000, 6000, 7000, 8000])
pl.imshow(np.log10(N.T), origin='lower',
          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
          aspect='auto', interpolation='nearest', cmap=cmap)
pl.xlim(xedges[-1], xedges[0])
pl.ylim(yedges[-1], yedges[0])
pl.xlabel(r'$\mathrm{T_{eff}}$')
pl.ylabel(r'$\mathrm{log(g)}$')

cb = pl.colorbar(ticks=[0, 1, 2, 3],
                 format=r'$10^{%i}$', orientation='horizontal')
cb.set_label(r'$\mathrm{number\ in\ pixel}$')
pl.clim(0, 3)

#--------------------
# Second axes:
pl.subplot(132, xticks=[4000, 5000, 6000, 7000, 8000])
pl.imshow(FeH_mean.T, origin='lower',
          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
          aspect='auto', interpolation='nearest', cmap=cmap)
pl.xlim(xedges[-1], xedges[0])
pl.ylim(yedges[-1], yedges[0])
pl.xlabel(r'$\mathrm{T_{eff}}$')

cb = pl.colorbar(ticks=np.arange(-2.5, 1, 0.5),
                 format=r'$%.1f$', orientation='horizontal')
cb.set_label(r'$\mathrm{mean\ [Fe/H]\ in\ pixel}$')
pl.clim(-2.5, 0.5)

# Draw density contours over the colors
levels = np.linspace(0, np.log10(N.max()), 7)[2:]
pl.contour(np.log10(N.T), levels, colors='k', linewidths=1,
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

#--------------------
# Third axes:
pl.subplot(133, xticks=[4000, 5000, 6000, 7000, 8000])
pl.imshow(FeH_mean.T, origin='lower',
          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
          aspect='auto', interpolation='nearest', cmap=cmap_multicolor)
pl.xlim(xedges[-1], xedges[0])
pl.ylim(yedges[-1], yedges[0])
pl.xlabel(r'$\mathrm{T_{eff}}$')

cb = pl.colorbar(ticks=np.arange(-2.5, 1, 0.5),
                 format=r'$%.1f$', orientation='horizontal')
cb.set_label(r'$\mathrm{mean\ [Fe/H]\ in\ pixel}$')
pl.clim(-2.5, 0.5)

# Draw density contours over the colors
levels = np.linspace(0, np.log10(N.max()), 7)[2:]
pl.contour(np.log10(N.T), levels, colors='k', linewidths=1,
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

pl.show()
