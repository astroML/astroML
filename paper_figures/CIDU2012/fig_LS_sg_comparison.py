"""
Generalized vs Standard Lomb-Scargle
------------------------------------
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure is an example from astroML: see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt

from astroML.time_series import lomb_scargle

#------------------------------------------------------------
# Generate data where y is positive
np.random.seed(0)
N = 30
P = 0.3

t = P / 2 * np.random.random(N) + P * np.random.randint(100, size=N)
y = 10 + np.sin(2 * np.pi * t / P)
dy = 0.5 + 0.5 * np.random.random(N)
y_obs = y + np.random.normal(dy)

omega_0 = 2 * np.pi / P

#------------------------------------------------------------
# Compute the Lomb-Scargle Periodogram
sig = np.array([0.1, 0.01, 0.001])
omega = np.linspace(17, 22, 1000)
P_S = lomb_scargle(t, y, dy, omega, generalized=False)
P_G, z = lomb_scargle(t, y, dy, omega, generalized=True, significance=sig)

#------------------------------------------------------------
# Plot the results
fig = plt.figure()

# First panel: input data
ax = fig.add_subplot(211)
ax.errorbar(t, y_obs, dy, fmt='.k', lw=1, ecolor='gray')
ax.plot([-2, 32], [10, 10], ':k', lw=1)

ax.set_xlim(-2, 32)
ax.set_xlabel('$t$')
ax.set_ylabel('$y(t)$')

# Second panel: periodogram
ax = fig.add_subplot(212)
ax.plot(omega, P_S, '--k', lw=1, label='standard')
ax.plot(omega, P_G, '-k', lw=1, label='generalized')
ax.legend(loc=2, prop=dict(size=14))

# plot the significance lines.
xlim = (omega[0], omega[-1])
for zi, pi in zip(z, sig):
    ax.plot(xlim, (zi, zi), ':k', lw=1)
    ax.text(xlim[-1] - 0.001, zi - 0.02, "$%.1g$" % pi, ha='right', va='top')

ax.set_xlabel('$\omega$')
ax.set_ylabel('$P_{LS}(\omega)$')
ax.set_ylim(0, 1.1)
plt.show()
