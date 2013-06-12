"""
Fourier Reconstruction of a Gaussian
------------------------------------
This figure demonstrates Fourier decomposition of a Gaussian
"""
# Author: Jake VanderPlas
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general
import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import norm

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=True)

x = np.linspace(-50, 50, 10000)
y = norm.pdf(x, 0, 1)

fig = plt.figure(figsize=(5, 3.75))
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
                transform=ax.transAxes)
    else:
        ax.text(0.01, 0.95, "%i modes" % k, ha='left', va='top',
                transform=ax.transAxes)

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
