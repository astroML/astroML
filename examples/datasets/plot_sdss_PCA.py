"""
============================================
Principal Component Analysis of SDSS Spectra
============================================

Plot the principal components and eigenvectors of 4000 spectra from the
Sloan Digital Sky Survey.
"""
import numpy as np
import pylab as pl

from astroML.plotting import multiscatter, multidensity
from astroML.datasets import\
    sdss_corrected_spectra,fetch_sdss_corrected_spectra

data = fetch_sdss_corrected_spectra()

spec_recons = sdss_corrected_spectra.reconstruct_spectra(data)
lam = sdss_corrected_spectra.compute_wavelengths(data)

mu = data['mu']
evecs = data['evecs']
coeffs = data['coeffs']

c = data['lineindex_cln']
c[c > 5] = 0

# Scatter coefficients
axlist = multiscatter(coeffs[:, :4], labels=('c1', 'c2', 'c3', 'c4'),
                      scatter_kwargs = dict(s=4, lw=0, c=c))


# Plot mean and principal components
pl.figure(figsize=(6, 10))
ax = pl.subplot(611, yticks=[])
pl.text(0.99, 0.95, 'mean', ha='right', va='top', transform=ax.transAxes) 
pl.plot(lam, mu)
for i in range(5):
    axi = pl.subplot(612 + i, sharex=ax, yticks=[])
    pl.plot(lam, evecs[i])
    ylim = pl.ylim()
    dy = ylim[1] - ylim[0]
    pl.ylim(ylim[0] - 0.05 * dy,
            ylim[1] + 0.05 * dy)
    pl.text(0.99, 0.95, 'c%i' % (i + 1),
            ha='right', va='top', transform=axi.transAxes) 
pl.xlim(3010, 7990)
pl.subplots_adjust(hspace=0)

pl.show()
