"""
Eddington-Malmquist & Lutz-Kelker Biases
----------------------------------------
These plots show the magnitude biases that result from the Eddington-Malmquist
and Lutz-Kelker Biases.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt
from astroML.stats import median_sigmaG


def generate_magnitudes(N, k=0.6, m_min=20, m_max=25):
    """
    generate magnitudes from a distribution with
      p(m) ~ 10^(k m)
    """
    klog10 = k * np.log(10)
    Pmin = np.exp(klog10 * m_min)
    Pmax = np.exp(klog10 * m_max)
    return (1. / klog10) * np.log(Pmin + (Pmax - Pmin) * np.random.random(N))


def mag_errors(m_true, m5=24.0, fGamma=0.039):
    """
    compute magnitude errors based on the true magnitude and the
    5-sigma limiting magnitude, m5
    """
    x = 10 ** (0.4 * (mtrue - m5))
    return np.sqrt((0.04 - fGamma) * x + fGamma * x ** 2)

#----------------------------------------------------------------------
# Compute the Eddington-Malmquist bias & scatter

mtrue = generate_magnitudes(1E6, m_min=20, m_max=25)
photomErr = mag_errors(mtrue)

m1 = mtrue + np.random.normal(0, photomErr)
m2 = mtrue + np.random.normal(0, photomErr)
dm = m1 - m2

mGrid = np.linspace(21, 24, 50)
medGrid = np.zeros(mGrid.size)
sigGrid = np.zeros(mGrid.size)

for i in range(mGrid.size):
    medGrid[i], sigGrid[i] = median_sigmaG(dm[m1 < mGrid[i]])

#----------------------------------------------------------------------
# Lutz-Kelker bias and scatter

mtrue = generate_magnitudes(1E6, m_min=17, m_max=20)
relErr = 0.3 * 10 ** (0.4 * (mtrue - 20))

pErrGrid = np.arange(0.02, 0.31, 0.01)

deltaM2 = 5 * np.log(1 + 2 * relErr ** 2)
deltaM4 = 5 * np.log(1 + 4 * relErr ** 2)

med2 = [np.median(deltaM2[relErr < e]) for e in pErrGrid]
med4 = [np.median(deltaM4[relErr < e]) for e in pErrGrid]

#----------------------------------------------------------------------
# plot results
fig = plt.figure(figsize=(8, 4))
fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25,
                    bottom=0.15, top=0.95)

ax = fig.add_subplot(121)
ax.plot(mGrid, sigGrid, '-k', label='scatter')
ax.plot(mGrid, medGrid, '--k', label='bias')
ax.plot(mGrid, 0 * mGrid, ':k', lw=1)
ax.legend(loc=2, prop=dict(size=14))

ax.set_xlabel(r'$m_{\rm obs}$')
ax.set_ylabel('bias/scatter (mag)')

ax.set_ylim(-0.04, 0.21)
ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))

for l in ax.yaxis.get_minorticklines():
    l.set_markersize(3)

ax = fig.add_subplot(122)
ax.plot(pErrGrid, med2, '-k', label='p=2')
ax.plot(pErrGrid, med4, '--k', label='p=4')
ax.legend(loc=2, prop=dict(size=14))
ax.set_xlabel(r'$\sigma_\pi / \pi$')
ax.set_ylabel('absolute magnitude bias')
ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.set_xlim(0.02, 0.301)
ax.set_ylim(0, 0.701)

plt.show()
