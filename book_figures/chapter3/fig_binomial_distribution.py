"""
Example of a Binomial distribution
----------------------------------

This shows an example of a binomial distribution with various parameters.
We'll generate the distribution using::

    dist = scipy.stats.binom(...)

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
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from scipy.stats import binom
from matplotlib import pyplot as plt

#------------------------------------------------------------
# Define the distribution parameters to be plotted
n_values = [20, 20, 40]
b_values = [0.2, 0.6, 0.6]
linestyles = ['-', '--', ':']
x = np.arange(200)

#------------------------------------------------------------
# plot the distributions
for (n, b, ls) in zip(n_values, b_values, linestyles):
    # create a binomial distribution
    dist = binom(n, b)

    plt.plot(x, dist.pmf(x), ls=ls, c='black',
             label=r'$b=%.1f,\ n=%i$' % (b, n), linestyle='steps')

plt.xlim(0, 40)
plt.ylim(0, 0.3)

plt.xlabel('$x$', fontsize=14)
plt.ylabel(r'$P(x|b, n)$', fontsize=14)
plt.title('Binomial Distribution')

plt.legend()
plt.show()
