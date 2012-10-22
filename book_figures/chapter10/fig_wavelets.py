"""
Examples of Wavelets
--------------------
This figure shows some examples of wavelets used in the wavelet transforms.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt

from astroML.fourier import FT_continuous, IFT_continuous, sinegauss

#------------------------------------------------------------
# Set up the wavelets
t0 = 0
t = np.linspace(-0.4, 0.4, 10000)
f0 = np.array([5, 5, 10, 10])
Q = np.array([1, 0.5, 1, 0.5])

# compute wavelets all at once
W = sinegauss(t, t0, f0[:, None], Q[:, None])

#------------------------------------------------------------
# Plot the wavelets
fig = plt.figure()
fig.subplots_adjust(hspace=0.05, wspace=0.05)

# in each panel, plot and label a different wavelet
for i in range(4):
    ax = fig.add_subplot(221 + i)
    ax.plot(t, W[i].real, '-k')
    ax.plot(t, W[i].imag, '--k')

    ax.text(0.02, 0.98, "$f_0 = %i$\n$Q = %.1f$" % (f0[i], Q[i]),
            ha='left', va='top', transform=ax.transAxes, size=14)

    ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(-0.35, 0.35)

    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))

    if i in (0, 1):
        ax.xaxis.set_major_formatter(plt.NullFormatter())
    else:
        ax.set_xlabel('$t$')

    if i in (1, 3):
        ax.yaxis.set_major_formatter(plt.NullFormatter())
    else:
        ax.set_ylabel('$w(t)$')

plt.show()
