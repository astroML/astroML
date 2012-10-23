"""
Plot the posterior of mu vs g1 with outliers
--------------------------------------------
This plot shows an example of dealing with outliers in a Bayesian fashion.
Here we plot the posterior marginalized over all parameters except the
mean, and the probability of a particular point being an outlier.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from astroML.plotting.mcmc import convert_to_stdev


def p(mu, g1, xi, sigma1, sigma2):
    """Equation 5.97: marginalized likelihood over outliers"""
    L = (g1 * norm.pdf(xi[0], mu, sigma1) +
         (1 - g1) * norm.pdf(xi[0], mu, sigma2))

    mu = mu.reshape(mu.shape + (1,))
    g1 = g1.reshape(g1.shape + (1,))

    return L * np.prod(norm.pdf(xi[1:], mu, sigma1)
                       + norm.pdf(xi[1:], mu, sigma2), -1)

#------------------------------------------------------------
# Sample the points
np.random.seed(138)

N1 = 8
N2 = 2
sigma1 = 1
sigma2 = 3

sigmai = np.zeros(N1 + N2)
sigmai[N2:] = sigma1
sigmai[:N2] = sigma2

xi = np.random.normal(0, sigmai)

#------------------------------------------------------------
# Compute the marginalized posterior for the first and last point
mu = np.linspace(-5, 5, 71)
g1 = np.linspace(0, 1, 11)

L1 = p(mu[:, None], g1, xi, 1, 10)
L1 /= np.max(L1)

L2 = p(mu[:, None], g1, xi[::-1], 1, 10)
L2 /= np.max(L2)

#------------------------------------------------------------
# Plot the results
fig = plt.figure(figsize=(8, 4))
fig.subplots_adjust(left=0.1, right=0.95, wspace=0.05,
                    bottom=0.15, top=0.9)

ax1 = fig.add_subplot(121)
ax1.imshow(L1.T, origin='lower', aspect='auto', cmap=plt.cm.binary,
           extent=[mu[0], mu[-1], g1[0], g1[-1]])
ax1.contour(mu, g1, convert_to_stdev(np.log(L1).T),
            levels=(0.683, 0.955, 0.997),
            colors='k', linewidths=2)
ax1.set_xlabel(r'$\mu$')
ax1.set_ylabel(r'$g_1$')

ax2 = fig.add_subplot(122)
ax2.imshow(L2.T, origin='lower', aspect='auto', cmap=plt.cm.binary,
           extent=[mu[0], mu[-1], g1[0], g1[-1]])
ax2.contour(mu, g1, convert_to_stdev(np.log(L2).T),
            levels=(0.683, 0.955, 0.997),
            colors='k', linewidths=2)
ax2.set_xlabel(r'$\mu$')
ax2.yaxis.set_major_locator(plt.NullLocator())

plt.show()
