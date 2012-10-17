"""
Phased LINEAR Light Curve
-------------------------
Plot the colors, magnitudes, and periods of the LINEAR variable stars,
as well as the phased light curve of a single LINEAR object.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning for Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
import pylab as pl
from astroML.datasets import fetch_LINEAR_sample

#------------------------------------------------------------
# Get data for the plot
data = fetch_LINEAR_sample()

# Compute the phased light curve for a single object.
# the best-fit period in the file is not accurate enough
# for light curve phasing.  The frequency below is
# calculated using Lomb Scargle (see chapter10/fig_LINEAR_LS.py)
id = 18525697
omega = 10.82722481
t, y, dy = data[id].T
phase = (t * omega * 0.5 / np.pi + 0.1) % 1

# Select colors, magnitudes, and periods from the global set
targets = data.targets[data.targets['LP1'] < 2]
r = targets['r']
gr = targets['gr']
ri = targets['ri']
logP = targets['LP1']

#------------------------------------------------------------
# plot the results
fig = pl.figure(figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1,
                    top=0.95, right=0.95)

ax = fig.add_axes((0.62, 0.62, 0.3, 0.25))
pl.errorbar(phase, y, dy, fmt='.', color='black', ecolor='gray', lw=1)
pl.ylim(pl.ylim()[::-1])
pl.xlabel('phase')
pl.ylabel('magnitude')
ax.yaxis.set_major_locator(pl.MultipleLocator(0.5))
pl.title("example of\nphased light curve", fontsize=14)

ax = fig.add_subplot(223)
ax.plot(gr, ri, '.', color='black', markersize=2)
ax.set_xlim(-1.5, 1.7)
ax.set_ylim(-1.0, 1.9)
ax.xaxis.set_major_locator(pl.MultipleLocator(1.0))
ax.yaxis.set_major_locator(pl.MultipleLocator(1.0))
ax.set_xlabel('g-r')
ax.set_ylabel('r-i')

ax = fig.add_subplot(221, yscale='log')
ax.plot(gr, 10 ** logP, '.', color='black', markersize=2)
ax.set_xlim(-1.5, 1.7)
ax.set_ylim(1E-2, 1E2)
ax.xaxis.set_major_locator(pl.MultipleLocator(1.0))
ax.xaxis.set_major_formatter(pl.NullFormatter())
ax.set_ylabel('principal period (days)')

ax = fig.add_subplot(224, xscale='log')
ax.plot(10 ** logP, ri, '.', color='black', markersize=2)
ax.set_xlim(1E-2, 1E2)
ax.set_ylim(-1.0, 1.9)
ax.yaxis.set_major_formatter(pl.NullFormatter())
ax.yaxis.set_major_locator(pl.MultipleLocator(1.0))
ax.set_xlabel('principal period (days)')

pl.show()
