"""
Linear Sum of Gaussians
-----------------------

Fitting a spectrum with a linear sum of gaussians.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning for Astronomy" (2013)
#   For more information, see http://astroML.github.com
import pylab as pl
from astroML.datasets import fetch_spectrum
from astroML.sum_of_norms import sum_of_norms, norm

# Fetch the data
x, y = fetch_spectrum()

for n_gaussians in (10, 50, 100):
    # compute the best-fit linear combination
    w_best, rms, locs, widths = sum_of_norms(x, y, n_gaussians,
                                             spacing='linear',
                                             full_output=True)

    norms = w_best * norm(x[:, None], locs, widths)

    # plot the results
    pl.figure()
    pl.plot(x, y, '-k', label='input spectrum')
    ylim = pl.ylim()

    pl.plot(x, norms, ls='-', c='#FFAAAA')
    pl.plot(x, norms.sum(1), '-r', label='sum of gaussians')
    pl.ylim(-0.1 * ylim[1], ylim[1])

    pl.legend(loc=0)

    pl.text(0.97, 0.8,
            "rms error = %.2g" % rms,
            ha='right', va='top', transform=pl.gca().transAxes)
    pl.title("Fit to a Spectrum with a Sum of %i Gaussians" % n_gaussians)

pl.show()
