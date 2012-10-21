"""
Gaussian/Gaussian distribution
------------------------------
This shows the resulting observed distribution when points drawn from a
gaussian also have gaussian error bars
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, anderson
from astroML.stats import mean_sigma, median_sigmaG

#------------------------------------------------------------
# Create distributions

# draw underlying points
np.random.seed(0)
Npts = 1E6
x = np.random.normal(loc=0, scale=1, size=Npts)

# add error for each point
e = 3 * np.random.random(Npts)
x += np.random.normal(0, e)

# compute anderson-darling test
A2, sig, crit = anderson(x)
print "anderson-darling A^2 = %.1f" % A2

# compute point statistics
mu_sample, sig_sample = mean_sigma(x, ddof=1)
med_sample, sigG_sample = median_sigmaG(x)

#------------------------------------------------------------
# plot the results
ax = plt.axes()
ax.hist(x, 100, histtype='stepfilled', alpha=0.2,
        lw=1.5, color='k', normed=True)

# plot the fitting normal curves
x_sample = np.linspace(-15, 15, 1000)
ax.plot(x_sample, norm(mu_sample, sig_sample).pdf(x_sample),
        '-k', lw=1.5, label='$\sigma$ fit')
ax.plot(x_sample, norm(med_sample, sigG_sample).pdf(x_sample),
        '--k', lw=1.5, label='$\sigma_G$ fit')
ax.legend()

ax.set_xlim(-10, 10)
ax.set_xlabel('x')
ax.set_ylabel('p(x)')
plt.show()
