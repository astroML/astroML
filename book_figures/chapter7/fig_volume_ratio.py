"""
Curse of Dimensionality: Volume Ratio
-------------------------------------

This figure shows the ratio of the volume of a unit hypercube to the volume
of an inscribed hypersphere.  The curse of dimensionality is illustrated in
the fact that this ratio approaches zero as the number of dimensions
approaches infinity.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning for Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
import pylab as pl
from scipy.special import gamma, gammaln

dims = np.arange(0, 51)

# log of volume of a sphere with r = 1
log_V_sphere = (np.log(2) + 0.5 * dims * np.log(np.pi)
                - np.log(dims) - gammaln(0.5 * dims))

log_V_cube = dims * np.log(2)

# compute the log of f_k to avoid overflow errors
log_f_k = log_V_sphere - log_V_cube

pl.semilogy(dims, np.exp(log_V_cube), '-k',
            label='side-2 hypercube')
pl.semilogy(dims, np.exp(log_V_sphere), '--k',
            label='inscribed unit hypersphere')

pl.xlim(0, 50)
pl.ylim(1E-13, 1E15)

pl.xlabel('Number of Dimensions')
pl.ylabel('Hyper-Volume')
pl.legend(loc=3)
pl.show()
