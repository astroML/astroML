"""
Fourier Reconstruction of a Gaussian
------------------------------------
This figure demonstrates Fourier decomposition of a Gaussian
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import norm

x = np.linspace(-50, 50, 10000)
y = norm.pdf(x, 0, 1)

fig = plt.figure()
fig.subplots_adjust(hspace=0)

kvals = [20, 30, 50]
subplots = [311, 312, 313]

for (k, subplot) in zip(kvals, subplots):
    ax = fig.add_subplot(subplot)

    # Use FFT to fit a truncated Fourier series
    y_fft = np.fft.fft(y)
    y_fft[k + 1:-k] = 0
    y_fit = np.fft.ifft(y_fft).real

    ax.plot(x, y, color='gray')
    ax.plot(x, y_fit, color='black')

    if k == 1:
        ax.text(0.01, 0.95, "1 mode", ha='left', va='top',
                fontsize=14, transform=ax.transAxes)
    else:
        ax.text(0.01, 0.95, "%i modes" % k, ha='left', va='top',
                fontsize=14, transform=ax.transAxes)

    if subplot == subplots[-1]:
        ax.set_xlabel('phase')
    else:
        ax.xaxis.set_major_formatter(plt.NullFormatter())

    if subplot == subplots[1]:
        ax.set_ylabel('amplitude')
    ax.yaxis.set_major_formatter(plt.NullFormatter())

    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.05, 0.5)


plt.show()
