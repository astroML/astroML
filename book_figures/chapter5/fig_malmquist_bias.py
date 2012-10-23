"""
Malmquist Bias Example
----------------------
This figure demonstrates the Malmquist bias, whereby observational cuts in
noisy data can lead to sets of observations which are statistically biased.
The left panel shows the true distribution, and the observed (noisy)
distribution.  The right panel shows the biased distribution of observed
values.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt
from astroML.stats.random import trunc_exp

#------------------------------------------------------------
# Sample from a truncated exponential distribution
N = 1E6
hmin = 4.3
hmax = 5.7
k = 0.6 * np.log(10)
true_dist = trunc_exp(hmin - 1.4,
                      hmax + 3.4,
                      0.6 * np.log(10))

# draw the true distributions and heteroscedastic noise
np.random.seed(0)
h_true = true_dist.rvs(N)
dh = 0.5 * (1 + np.random.random(N))
h_obs = np.random.normal(h_true, dh)

# create observational cuts
cut = (h_obs < hmax) & (h_obs > hmin)

# select a random (not observationally cut) subsample
rand = np.arange(len(h_obs))
np.random.shuffle(rand)
rand = rand[:cut.sum()]

#------------------------------------------------------------
# plot the results
fig = plt.figure(figsize=(8, 4))
fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25,
                    bottom=0.15, top=0.9)

# First axes: plot the true and observed distribution
ax = fig.add_subplot(121)
bins = np.linspace(0, 12, 100)

x_pdf = np.linspace(0, 12, 1000)
ax.plot(x_pdf, true_dist.pdf(x_pdf), '-k', lw=2,
        label='true distribution')
ax.hist(h_obs, bins, histtype='stepfilled',
        alpha=0.3, fc='b', normed=True,
        label='observed distribution')
ax.legend(loc=2, prop=dict(size=12))

ax.add_patch(plt.Rectangle((hmin, 0), hmax - hmin, 1.2,
                           fc='gray', ec='k', linestyle='dashed',
                           alpha=0.3))
ax.text(5, 0.07, 'sampled region', rotation=45, ha='center', va='center',
        color='gray', fontsize=12)

ax.set_xlim(hmin - 1.3, hmax + 1.3)
ax.set_ylim(0, 0.14001)
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.set_xlabel(r'$x_{\rm obs}$')
ax.set_ylabel(r'$p(x_{\rm obs})$')

# Second axes: plot the histogram of (x_obs - x_true)
ax = fig.add_subplot(122)
bins = 30
ax.hist(h_obs[cut] - h_true[cut], bins, histtype='stepfilled',
        alpha=0.3, color='k', normed=True, label='observed\nsample')
ax.hist(h_obs[rand] - h_true[rand], bins, histtype='step',
        color='k', linestyle='dashed', normed=True, label='random\nsample')
ax.plot([0, 0], [0, 1], ':k', lw=1)
ax.legend(prop=dict(size=12), ncol=2, loc='upper center', frameon=False)

ax.set_xlim(-4, 4)
ax.set_ylim(0, 0.65)
ax.set_xlabel(r'$x_{\rm obs} - x_{\rm true}$')
ax.set_ylabel(r'$p(x_{\rm obs} - x_{\rm true})$')
plt.show()
