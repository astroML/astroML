"""
Selection of Histogram bin size
-------------------------------

This shows an example of selecting histogram bin size using Scott's rule,
the Freedman-Diaconis rule, and Knuth's rule.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning for Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
import pylab as pl
from scipy import stats
from astroML.density_estimation import \
    scotts_bin_width, freedman_bin_width, knuth_nbins


def plot_labeled_histogram(style, data,
                           x, pdf_true, ax=None,
                           hide_x=False,
                           hide_y=False):
    if style == 'scott':
        name = 'Scott\'s Rule'
        dx = scotts_bin_width(data)
        Nbins = int((data.max() - data.min()) / dx)
    elif style == 'freedman':
        name = 'Freedman-Diaconis'
        dx = freedman_bin_width(data)
        Nbins = int((data.max() - data.min()) / dx)
    elif style == 'knuth':
        name = 'Knuth\'s Rule'
        Nbins, dx = knuth_nbins(data, True)
    else:
        raise ValueError

    print '%s: %i bins' % (name, Nbins)

    if ax is not None:
        ax = pl.axes(ax)

    pl.hist(data, Nbins, color='k', histtype='step', normed=True)
    pl.text(0.99, 0.95, '%s:\n%i bins' % (name, Nbins),
            transform=ax.transAxes,
            ha='right', va='top', fontsize=12)

    ax.fill(x, pdf_true, '-', color='#CCCCCC', zorder=0)

    if hide_x:
        ax.xaxis.set_major_formatter(pl.NullFormatter())
    if hide_y:
        ax.yaxis.set_major_formatter(pl.NullFormatter())

    ax.set_xlim(-5, 5)

    return ax


#------------------------------------------------------------
# Set up distributions:
Npts = 5000
np.random.seed(0)
x = np.linspace(-6, 6, 1000)

# Gaussian distribution
data_G = stats.norm(0, 1).rvs(Npts)
pdf_G = stats.norm(0, 1).pdf(x)

# Non-Gaussian distribution
distributions = [stats.laplace(0, 0.4),
                 stats.norm(-4.0, 0.2),
                 stats.norm(4.0, 0.2)]

weights = np.array([0.8, 0.1, 0.1])
weights /= weights.sum()

data_NG = np.hstack(d.rvs(int(w * Npts))
                    for (d, w) in zip(distributions, weights))
pdf_NG = sum(w * d.pdf(x)
             for (d, w) in zip(distributions, weights))

#------------------------------------------------------------
# Plot results
fig = pl.figure(figsize=(10, 5))
fig.subplots_adjust(hspace=0, left=0.05, right=0.95, wspace=0.05)
ax = [fig.add_subplot(3, 2, i + 1) for i in range(6)]

# first column: Gaussian distribution
plot_labeled_histogram('scott', data_G, x, pdf_G,
                       ax=ax[0], hide_x=True, hide_y=True)
plot_labeled_histogram('freedman', data_G, x, pdf_G,
                       ax=ax[2], hide_x=True, hide_y=True)
plot_labeled_histogram('knuth', data_G, x, pdf_G,
                       ax=ax[4], hide_x=False, hide_y=True)

ax[0].set_title('Gaussian distribution')
ax[2].set_ylabel('P(x)')
ax[4].set_xlabel('x')

# second column: non-gaussian distribution
plot_labeled_histogram('scott', data_NG, x, pdf_NG,
                       ax=ax[1], hide_x=True, hide_y=True)
plot_labeled_histogram('freedman', data_NG, x, pdf_NG,
                       ax=ax[3], hide_x=True, hide_y=True)
plot_labeled_histogram('knuth', data_NG, x, pdf_NG,
                       ax=ax[5], hide_x=False, hide_y=True)

ax[1].set_title('non-Gaussian distribution')
ax[5].set_xlabel('x')

pl.show()
