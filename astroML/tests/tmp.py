import numpy as np

from astroML.resample import jackknife, jackknife_old

x = np.random.normal(0, 1, 1000)
mu, sig = jackknife(x, np.mean, kwargs=dict(axis=1))
print mu, sig
print
mu, sig = jackknife_old(x, np.mean, kwargs=dict(axis=1))
print mu, sig
