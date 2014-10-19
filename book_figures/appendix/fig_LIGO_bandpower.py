"""
Plot the band power of the LIGO big dog event
---------------------------------------------
"""
# Author: Jake VanderPlas
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general
from __future__ import division

import numpy as np
from matplotlib import pyplot as plt

from astroML.datasets import fetch_LIGO_bigdog
from astroML.fourier import FT_continuous

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=True)


def multiple_power_spectrum(t, x, window_size=10000, step_size=1000):
    assert x.shape == t.shape
    assert x.ndim == 1
    assert len(x) > window_size
    N_steps = (len(x) - window_size) // step_size

    indices = np.arange(window_size) + step_size * np.arange(N_steps)[:, None]
    X = x[indices].astype(complex)

    f, H = FT_continuous(t[:window_size], X)

    i = (f > 0)
    return f[i], abs(H[:, i])


X = fetch_LIGO_bigdog()
t = X['t']
x = X['Hanford']

window_size = 10000
step_size = 500

f, P = multiple_power_spectrum(t, x,
                               window_size=window_size,
                               step_size=step_size)

i = (f > 50) & (f < 1500)
P = P[:, i]
f = f[i]

fig = plt.figure(figsize=(5, 3.75))
plt.imshow(np.log10(P).T, origin='lower', aspect='auto',
           extent=[t[window_size / 2],
                   t[window_size / 2 + step_size * P.shape[0]],
                   f[0], f[-1]])
plt.xlabel('t (s)')
plt.ylabel('f (Hz) derived from %.2fs window' % (t[window_size] - t[0]))
plt.colorbar().set_label('$|H(f)|$')
plt.show()
