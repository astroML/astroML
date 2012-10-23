"""
Example of a Cauchy distribution
--------------------------------

This shows an example of a Cauchy distribution with various parameters.
We'll generate the distribution using::

    dist = scipy.stats.cauchy(...)

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
from scipy.stats import cauchy
from matplotlib import pyplot as plt

#------------------------------------------------------------
# Define the distribution parameters to be plotted
gamma_values = [0.5, 1.0, 2.0]
linestyles = ['-', '--', ':']
mu = 0
x = np.linspace(-10, 10, 1000)

#------------------------------------------------------------
# plot the distributions
for gamma, ls in zip(gamma_values, linestyles):
    dist = cauchy(mu, gamma)

    plt.plot(x, dist.pdf(x), ls=ls, color='black',
             label=r'$\mu=%i,\ \gamma=%.1f$' % (mu, gamma))

plt.xlim(-5, 5)
plt.ylim(0, 0.8)

plt.xlabel('$x$', fontsize=14)
plt.ylabel(r'$P(x|\mu,\gamma)$', fontsize=14)
plt.title('Cauchy Distribution')

plt.legend()
plt.show()
