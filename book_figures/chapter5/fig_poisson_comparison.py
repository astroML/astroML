"""
Poisson Statistics with arbitrarily small bins
----------------------------------------------
This figure compares the binned and non-binned approach to regression
in the Poissonian context.  As expected, when the number of bins becomes
very large, the non-binned and binned cases lead to similar results.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt
from astroML.stats.random import linear


def logL_continuous(x, a, xmin, xmax):
    """Continuous log-likelihood (Eq. 5.84)"""
    x = x.ravel()
    a = a.reshape(a.shape + (1,))

    mu = 0.5 * (xmin + xmax)
    W = (xmax - xmin)

    return np.sum(np.log(a * (x - mu) + 1. / W), -1)


def logL_poisson(xi, yi, a, b):
    """poisson log-likelihood (Eq. 5.88)"""
    xi = xi.ravel()
    yi = yi.ravel()
    a = a.reshape(a.shape + (1,))
    b = b.reshape(b.shape + (1,))
    yyi = a * xi + b
    return np.sum(yi * np.log(yyi) - yyi, -1)

#----------------------------------------------------------------------
# draw the data
np.random.seed(0)
N = 200
a_true = 0.01
xmin = 0.0
xmax = 10.0

lin_dist = linear(xmin, xmax, a_true)
x = lin_dist.rvs(N)

a = np.linspace(0.0, 0.02, 101)
b = np.linspace(0.00001, 0.15, 51)

#------------------------------------------------------------
# Compute the log-likelihoods

# continuous case
logL = logL_continuous(x, a, xmin, xmax)
L_c = np.exp(logL - logL.max())
L_c /= L_c.sum() * (a[1] - a[0])

# discrete case: compute for 2 and 1000 bins
nbins = [1000, 2]
L_p = [0, 0]

for i, n in enumerate(nbins):
    yi, bins = np.histogram(x, bins=np.linspace(xmin, xmax, n + 1))
    xi = 0.5 * (bins[:-1] + bins[1:])

    factor = N * (xmax - xmin) * 1. / n

    logL = logL_poisson(xi, yi, factor * a, factor * b[:, None])
    L_p[i] = np.exp(logL - np.max(logL)).sum(0)
    L_p[i] /= L_p[i].sum() * (a[1] - a[0])


#------------------------------------------------------------
# Plot the results
ax = plt.axes()

ax.plot(a, L_c, '-k', label='continuous')
for L, ls, n in zip(L_p, ['-', '--'], nbins):
    ax.plot(a, L, ls, color='gray', lw=1,
            label='discrete, %i bins' % n)

# plot a vertical line: in newer matplotlib, use ax.vlines([a_true])
ylim = (0, 200)
plt.plot([a_true, a_true], ylim, ':k', lw=1)

ax.set_xlim(0, 0.02)
ax.set_ylim(ylim)

ax.set_xlabel('$a$')
ax.set_ylabel('$p(a)$')

ax.legend(loc=2, prop=dict(size=14))

plt.show()
