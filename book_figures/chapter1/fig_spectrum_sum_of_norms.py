"""
Linear Sum of Gaussians
-----------------------

Fitting a spectrum with a linear sum of gaussians.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
from matplotlib import pyplot as plt
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
    plt.figure()
    plt.plot(x, y, '-k', label='input spectrum')
    ylim = plt.ylim()

    plt.plot(x, norms, ls='-', c='#FFAAAA')
    plt.plot(x, norms.sum(1), '-r', label='sum of gaussians')
    plt.ylim(-0.1 * ylim[1], ylim[1])

    plt.legend(loc=0)

    plt.text(0.97, 0.8,
            "rms error = %.2g" % rms,
            ha='right', va='top', transform=plt.gca().transAxes)
    plt.title("Fit to a Spectrum with a Sum of %i Gaussians" % n_gaussians)

plt.show()
