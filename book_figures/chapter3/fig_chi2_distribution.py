"""
Example of a chi-squared distribution
---------------------------------------

This shows an example of a :math:`\chi^2` distribution with various parameters.
We'll generate the distribution using::

    dist = scipy.stats.chi2(...)

Where ... should be filled in with the desired distribution parameters
Once we have defined the distribution parameters in this way, these
distribution objects have many useful methods; for example:

* ``dist.pmf(x)`` computes the Probability Mass Function at values ``x``
  in the case of discrete distributions

* ``dist.pdf(x)`` computes the Probability Density Function at values ``x``
  in the case of continuous distributions

* ``dist.rvs(N)`` computes ``N`` random variables distributed according
  to the given distribution

Many further options exist; refer to the documentation of ``scipy.stats``
for more details.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning for Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from scipy.stats import chi2
import pylab as pl

#------------------------------------------------------------
# Define the distribution parameters to be plotted
k_values = [1, 2, 5, 7]
linestyles = ['-', '--', ':', '-.']
mu = 0
x = np.linspace(-1, 20, 1000)

for k, ls in zip(k_values, linestyles):
    dist = chi2(k, mu)

    pl.plot(x, dist.pdf(x), ls=ls, c='black',
            label=r'$k=%i$' % k)

pl.xlim(0, 10)
pl.ylim(0, 0.6)

pl.xlabel('$Q$', fontsize=14)
pl.ylabel(r'$P(Q|k)$', fontsize=14)
pl.title(r'$\chi^2\ \mathrm{Distribution}$', fontsize=14)

pl.legend()
pl.show()
