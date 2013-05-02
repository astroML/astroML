"""
Phased LINEAR Light Curve
-------------------------
Plot the colors, magnitudes, and periods of the LINEAR variable stars,
as well as the phased light curve of a single LINEAR object.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt
from astroML.datasets import fetch_LINEAR_sample, fetch_LINEAR_geneva

#------------------------------------------------------------
# Get data for the plot
data = fetch_LINEAR_sample()
geneva = fetch_LINEAR_geneva()  # contains well-measured periods

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

# Cross-match by ID with the geneva catalog to get more accurate periods
targetIDs = map(lambda ID: str(ID).lstrip('0'), targets['objectID'])
genevaIDs = map(lambda ID: str(ID).lstrip('0'), geneva['LINEARobjectID'])

def safe_index(L, val):
    try:
        return L.index(val)
    except ValueError:
        return -1

ind = np.array([safe_index(genevaIDs, ID) for ID in targetIDs])
mask = (ind >= 0)

logP = geneva['logP'][ind[mask]]
r = r[mask]
gr = gr[mask]
ri = ri[mask]

#------------------------------------------------------------
# plot the results
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1,
                    top=0.95, right=0.95)

ax = fig.add_axes((0.62, 0.62, 0.3, 0.25))
plt.errorbar(phase, y, dy, fmt='.', color='black', ecolor='gray', lw=1)
plt.ylim(plt.ylim()[::-1])
plt.xlabel('phase')
plt.ylabel('magnitude')
ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.title("example of\nphased light curve", fontsize=14)

ax = fig.add_subplot(223)
ax.plot(gr, ri, '.', color='black', markersize=2)
ax.set_xlim(-1.5, 1.7)
ax.set_ylim(-1.0, 2.0)
ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
ax.set_xlabel(r'${\rm g-r}$')
ax.set_ylabel(r'${\rm r-i}$')

ax = fig.add_subplot(221, yscale='log')
ax.plot(gr, 10 ** logP, '.', color='black', markersize=2)
ax.set_xlim(-1.5, 1.7)
ax.set_ylim(3E-2, 1E1)
ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.set_ylabel('principal period (days)')

ax = fig.add_subplot(224, xscale='log')
ax.plot(10 ** logP, ri, '.', color='black', markersize=2)
ax.set_xlim(3E-2, 1E1)
ax.set_ylim(-1.0, 2.0)
ax.yaxis.set_major_formatter(plt.NullFormatter())
ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
ax.set_xlabel('principal period (days)')

plt.show()
