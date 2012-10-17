"""
========================================
WMAP power spectrum analysis with HealPy
========================================

This demonstrates how to plot and take a power spectrum of the WMAP data
using healpy, the python wrapper for healpix.

"""

import numpy as np
import pylab as pl

# warning: due to a bug in healpy, importing it before pylab can cause
#  a segmentation fault in some circumstances.
import healpy as hp

from astroML.datasets import fetch_wmap_temperatures


wmap_unmasked = fetch_wmap_temperatures(masked=False)
wmap_masked = fetch_wmap_temperatures(masked=True)
white_noise = np.ma.asarray(np.random.normal(0, 0.062, wmap_masked.shape))

# plot the unmasked map
fig = pl.figure(1)
hp.mollview(wmap_unmasked, min=-1, max=1, title='Unmasked map',
            fig=1, unit=r'$\Delta$T (mK)')

# plot the masked map
#  filled() fills the masked regions with a null value.
fig = pl.figure(2)
hp.mollview(wmap_masked.filled(), title='Masked map',
            fig=2, unit=r'$\Delta$T (mK)')

# compute and plot the power spectrum\
cl = hp.anafast(wmap_masked.filled(), lmax=1024)
ell = np.arange(len(cl))

cl_white = hp.anafast(white_noise, lmax=1024)

pl.figure()
pl.scatter(ell, ell * (ell + 1) * cl, s=4, c='black', lw=0)
pl.scatter(ell, ell * (ell + 1) * cl_white, s=4, c='gray', lw=0)
pl.xlabel(r'$\ell$')
pl.ylabel(r'$\ell(\ell+1)C_\ell$')
pl.title('Angular Power (not mask corrected)')
pl.grid()
pl.xlim(0, 1100)

pl.show()
