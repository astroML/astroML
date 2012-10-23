"""
Comparison of Lomb-Scargle Methods
----------------------------------
This shows a comparison of the Lomb-Scargle periodogram
and the Modified Lomb-Scargle periodogram for a single star,
along with the multi-term results.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt

from astroML.time_series import\
    lomb_scargle, search_frequencies, multiterm_periodogram
from astroML.datasets import fetch_LINEAR_sample

#id, period = 11375941, 58.4
id, period = 18525697, 17.05

data = fetch_LINEAR_sample()
t, y, dy = data[id].T

#omega, power = search_frequencies(t, y, dy)
#period = omega[np.argmax(power)]
#print period
#exit()

omega = np.linspace(period, period + 0.1, 1000)
ax = plt.subplot(211)
for n_terms in [1, 2, 3]:
    P1 = multiterm_periodogram(t, y, dy, omega, n_terms=n_terms)
    plt.plot(omega, P1, lw=1, label='m = %i' % n_terms)
plt.legend(loc=2)
plt.xlim(period, period + 0.1)
plt.ylim(0, 1.0)
plt.ylabel('$1 - \chi^2(\omega) / \chi^2_{ref}$')

plt.subplot(212, sharex=ax)
for generalized in [True, False]:
    if generalized:
        label = 'generalized LS'
    else:
        label = 'standard LS'
    P2 = lomb_scargle(t, y, dy, omega, generalized=generalized)
    plt.plot(omega, P2, lw=1, label=label)
plt.legend(loc=2)
plt.xlim(period, period + 0.1)
plt.ylim(0, 1.0)

plt.xlabel('frequency $\omega$')
plt.ylabel('$P_{LS}(\omega)$')

plt.show()
