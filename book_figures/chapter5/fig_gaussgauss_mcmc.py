"""
Gaussian Distribution with Gaussian Errors
------------------------------------------

This plot shows the Likelihood as a function of the mean :math:`\mu` and the
error :math:`\sigma` when the posterior is assumed to be gaussian, and the
points have heteroscedatic gaussian errors.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt

# Hack to fix import issue in older versions of pymc
import scipy
import scipy.misc
scipy.derivative = scipy.misc.derivative
import pymc

from astroML.plotting.mcmc import convert_to_stdev
from astroML.plotting import plot_mcmc


def gaussgauss_logL(xi, ei, mu, sigma):
    """Equation 5.22: gaussian likelihood"""
    ndim = len(np.broadcast(sigma, mu).shape)

    xi = xi.reshape(xi.shape + tuple(ndim * [1]))
    ei = ei.reshape(ei.shape + tuple(ndim * [1]))

    s2_e2 = sigma ** 2 + ei ** 2
    return -0.5 * np.sum(np.log(s2_e2) + (xi - mu) ** 2 / s2_e2, 0)


#------------------------------------------------------------
# Select the data
np.random.seed(5)
mu_true = 1.
sigma_true = 1.
N = 10
ei = 3 * np.random.random(N)
xi = np.random.normal(mu_true, np.sqrt(sigma_true ** 2 + ei ** 2))

#----------------------------------------------------------------------
# Set up MCMC for our model parameters: (mu, sigma, ei)
mu = pymc.Uniform('mu', -10, 10, value=0)
log_sigma = pymc.Uniform('log_sigma', -10, 10, value=0)
log_error = pymc.Uniform('log_error', -10, 10, value=np.zeros(N))


@pymc.deterministic
def sigma(log_sigma=log_sigma):
    return np.exp(log_sigma)


@pymc.deterministic
def error(log_error=log_error):
    return np.exp(log_error)


def gaussgauss_like(x, mu, sigma, error):
    """likelihood of gaussian with gaussian errors"""
    sig2 = sigma ** 2 + error ** 2
    x_mu2 = (x - mu) ** 2
    return -0.5 * np.sum(np.log(sig2) + x_mu2 / sig2)

GaussGauss = pymc.stochastic_from_dist('gaussgauss',
                                       logp=gaussgauss_like,
                                       dtype=np.float,
                                       mv=True)
M = GaussGauss('M', mu, sigma, error, observed=True, value=xi)

model = dict(mu=mu, log_sigma=log_sigma, sigma=sigma,
             log_error=log_error, error=error, M=M)

#------------------------------------------------------------
# perform the MCMC sampling
pymc.numpy.random.seed(0)
S = pymc.MCMC(model)
S.sample(iter=25000, burn=2000)

#------------------------------------------------------------
# Extract the MCMC traces
trace_mu = S.trace('mu')[:]
trace_sigma = S.trace('sigma')[:]

fig = plt.figure()
ax, = plot_mcmc([trace_mu, trace_sigma], fig=fig,
                limits=[(-3, 5), (0, 5)],
                labels=(r'$\mu$', r'$\sigma$'),
                levels=[0.683, 0.955, 0.997],
                colors='k', linewidths=2)

#----------------------------------------------------------------------
# Compute and plot likelihood with known ei for comparison
# (Same as fig_likelihood_gaussgauss)
sigma = np.linspace(0.01, 5, 41)
mu = np.linspace(-3, 5, 41)

logL = gaussgauss_logL(xi, ei, mu, sigma[:, np.newaxis])
logL -= logL.max()

im = ax.contourf(mu, sigma, convert_to_stdev(logL),
                 levels=(0, 0.683, 0.955, 0.997),
                 cmap=plt.cm.binary_r, alpha=0.5)
im.set_clim(0, 1.1)

ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'$\sigma$')
ax.set_xlim(-3, 5)
ax.set_ylim(0, 5)

ax.set_aspect(1. / ax.get_data_ratio())

plt.show()
