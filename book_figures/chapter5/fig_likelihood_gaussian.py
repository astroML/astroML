"""
Log-likelihood for Gaussian Distribution
----------------------------------------

This plot shows the Likelihood as a function of the mean :math:`\mu` and the
error :math:`\sigma` when the posterior is assumed to be gaussian.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
import pylab as pl
from astroML.plotting.likelihood import convert_to_stdev


def gauss_logL(xbar, V, n, sigma, mu):
    """Equation 5.57: gaussian likelihood"""
    return (-(n + 1) * np.log(sigma)
            - 0.5 * n * ((xbar - mu) ** 2 + V) / sigma ** 2)

#------------------------------------------------------------
# Define the grid and compute logL
sigma = np.linspace(1, 5, 70)
mu = np.linspace(-3, 5, 70)
xbar = 1
V = 4
n = 10

logL = gauss_logL(xbar, V, n, sigma[:, np.newaxis], mu)
logL -= logL.max()

#------------------------------------------------------------
# Plot the results
pl.imshow(logL, origin='lower',
          extent=(mu[0], mu[-1], sigma[0], sigma[-1]),
          cmap=pl.cm.binary,
          aspect='auto')
pl.colorbar()
pl.clim(-5, 0)

pl.contour(mu, sigma, convert_to_stdev(logL),
           levels=(0.683, 0.955, 0.997),
           colors='k', linewidths=2)

pl.text(0.5, 0.9, r'$L(\mu,\sigma)\ \mathrm{for\ \bar{x}=1,\ V=4,\ n=10}$',
        fontsize=18, bbox=dict(ec='k', fc='w', alpha=0.9),
        ha='center', va='center', transform=pl.gca().transAxes)

pl.xlabel(r'$\mu$')
pl.ylabel(r'$\sigma$')

pl.show()
