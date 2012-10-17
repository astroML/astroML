"""
Plot the band power of the LIGO big dog event
---------------------------------------------
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning for Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
import pylab as pl

from astroML.datasets import fetch_LIGO_bigdog
from astroML.fourier import FT_continuous


def multiple_power_spectrum(t, x, window_size=10000, step_size=1000):
    assert x.shape == t.shape
    assert x.ndim == 1
    assert len(x) > window_size
    N_steps = (len(x) - window_size) / step_size

    indices = np.arange(window_size) + step_size * np.arange(N_steps)[:, None]
    X = x[indices].astype(complex)

    #kernel = np.exp(-0.5 * np.linspace(-5, 5, window_size)**2)
    #X *= kernel

    f, H = FT_continuous(t[:window_size], X)

    i = (f > 0)
    return f[i], abs(H[:, i])


X = fetch_LIGO_bigdog()
t = X['t']
x = X['Hanford']

window_size=10000
step_size=500

f, P = multiple_power_spectrum(t, x,
                               window_size=window_size, step_size=step_size)

i = (f > 50) & (f < 1500)
#i = (f > 330) & (f < 360)
P = P[:, i]
f = f[i]

#for i in range(P.shape[0]):
#    pl.semilogy(f, P[i] * f, lw=1)
#pl.show()
#exit()

pl.imshow(np.log10(P).T, origin='lower', aspect='auto',
          extent=[t[window_size / 2],
                  t[window_size / 2 + step_size * P.shape[0]],
                  f[0], f[-1]])
pl.xlabel('t (s)')
pl.ylabel('f (Hz) derived from %.2fs window' % (t[window_size] - t[0]))
pl.colorbar().set_label('|H(f)|')
pl.show()
