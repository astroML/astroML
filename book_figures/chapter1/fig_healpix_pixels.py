"""
Demonstration of HEALPix pixelization
-------------------------------------
This uses HEALpy, the python wrapper for HEALpix, to plot the HEALPix
pixellization of the sky.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt
# warning: due to a bug in healpy, importing it before pylab can cause
#  a segmentation fault in some circumstances.
import healpy as hp

#------------------------------------------------------------
# Prepare the healpix pixels
NSIDE = 4
m = np.arange(hp.nside2npix(NSIDE))
print "number of pixels:", len(m)

#------------------------------------------------------------
# Plot the pixelization
fig = plt.figure(1)
hp.mollview(m, nest=True, title="HEALPix Pixels (Mollweide)",
            cmap=plt.cm.binary, fig=1)

# remove colorbar: we don't need it for this plot
fig.axes.remove(fig.axes[1])

plt.show()
