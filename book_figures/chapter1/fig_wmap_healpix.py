"""
WMAP plotting with HEALPix
--------------------------
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning for Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
import pylab as pl

# warning: due to a bug in healpy, importing it before pylab can cause
#  a segmentation fault in some circumstances.
import healpy as hp

from astroML.datasets import fetch_wmap_temperatures

#------------------------------------------------------------
# Fetch the wmap data
wmap_unmasked = fetch_wmap_temperatures(masked=False)

#------------------------------------------------------------
# plot the unmasked map
fig = pl.figure(1)
hp.mollview(wmap_unmasked, min=-1, max=1, title='Raw WMAP data',
            fig=1, cmap=pl.cm.jet, unit=r'$\Delta$T (mK)')
pl.show()
