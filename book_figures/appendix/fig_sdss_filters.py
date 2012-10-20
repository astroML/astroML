"""
SDSS Filters
------------
Download and plot the five SDSS filter bands along with a Vega spectrum.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning for Astronomy" (2013)
#   For more information, see http://astroML.github.com
import pylab as pl
from astroML.datasets import fetch_sdss_filter, fetch_vega_spectrum

#------------------------------------------------------------
# Set up figure and axes
fig = pl.figure()
ax = fig.add_subplot(111)

#----------------------------------------------------------------------
# Fetch and plot the Vega spectrum
spec = fetch_vega_spectrum()
lam = spec[0]
spectrum = spec[1] / 2.1 / spec[1].max()
ax.plot(lam, spectrum, '-k', lw=2)

#------------------------------------------------------------
# Fetch and plot the five filters
text_kwargs = dict(fontsize=20, ha='center', va='center', alpha=0.5)

for f, c, loc in zip('ugriz', 'bgrmk', [3500, 4600, 6100, 7500, 8800]):
    data = fetch_sdss_filter(f)
    ax.fill(data[0], data[1], ec=c, fc=c, alpha=0.4)
    ax.text(loc, 0.02, f, color=c, **text_kwargs)

ax.set_xlim(3000, 11000)

ax.set_title('SDSS Filters and Reference Spectrum')
ax.set_xlabel('Wavelength (Angstroms)')
ax.set_ylabel('normalized flux / filter transmission')

pl.show()