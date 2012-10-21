"""
Bayesian Blocks for Histograms
------------------------------
.. currentmodule:: astroML

Bayesian Blocks is a dynamic histogramming method which optimizes one of
several possible fitness functions to determine an optimal binning for
data, where the bins are not necessarily uniform width.  The astroML
implementation is based on [1]_.  For more discussion of this technique,
see the blog post at [2]_.

The code below uses a fitness function suitable for event data with possible
repeats.  More fitness functions are available: see :mod:`density_estimation`

References
~~~~~~~~~~
.. [1] Scargle, J `et al.` (2012)
       http://adsabs.harvard.edu/abs/2012arXiv1207.5578S
.. [2] http://jakevdp.github.com/blog/2012/09/12/dynamic-programming-in-python/
"""
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from astroML.plotting import hist

# draw a set of variables
np.random.seed(0)
t = np.concatenate([stats.cauchy(-5, 1.8).rvs(500),
                    stats.cauchy(-4, 0.8).rvs(2000),
                    stats.cauchy(-1, 0.3).rvs(500),
                    stats.cauchy(2, 0.8).rvs(1000),
                    stats.cauchy(4, 1.5).rvs(500)])

# truncate values to a reasonable range
t = t[(t > -15) & (t < 15)]

# plot a standard histogram in the background, with alpha transparency
hist(t, bins=200, histtype='stepfilled',
     alpha=0.2, normed=True, label='standard histogram')

# plot an adaptive-width histogram on top
hist(t, bins='blocks', color='black',
     histtype='step', normed=True, label='bayesian blocks')

plt.legend()
plt.xlabel('t')
plt.ylabel('P(t)')
plt.show()
