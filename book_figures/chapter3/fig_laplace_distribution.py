"""
Example of a Laplace distribution
----------------------------------

This shows an example of a Laplace distribution with various parameters.
We'll generate the distribution using::

    dist = scipy.stats.laplace(...)

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
from scipy.stats import laplace
from matplotlib import pyplot as plt

# Adjust font sizes for text
import matplotlib
matplotlib.rc('font', size=8)

#------------------------------------------------------------
# Define the distribution parameters to be plotted
delta_values = [0.5, 1.0, 2.0]
linestyles = ['-', '--', ':']
mu = 0
x = np.linspace(-10, 10, 1000)

#------------------------------------------------------------
# plot the distributions
fig, ax = plt.subplots(figsize=(5, 3.75))

for delta, ls in zip(delta_values, linestyles):
    dist = laplace(mu, delta)

    plt.plot(x, dist.pdf(x), ls=ls, c='black',
             label=r'$\mu=%i,\ \Delta=%.1f$' % (mu, delta))

plt.xlim(-6, 6)
plt.ylim(0, 1.0)

plt.xlabel('$x$')
plt.ylabel(r'$p(x|\mu,\Delta)$')
plt.title('Laplace Distribution')

plt.legend()
plt.show()
