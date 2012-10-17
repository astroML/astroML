"""
SDSS spectra ICA
----------------

Plot the ICA compotnents of the SDSS spectra
"""
import numpy as np
import pylab as pl
from matplotlib import ticker

from sklearn.decomposition import NMF

from astroML.datasets import\
    sdss_corrected_spectra, fetch_sdss_corrected_spectra

data = fetch_sdss_corrected_spectra()

spectra = data['spectra']

# NMF can't handle negative fluxes
spectra[spectra < 0] = 0

lam = sdss_corrected_spectra.compute_wavelengths(data)

spectra = spectra[:100]

nmf = NMF(5, sparseness='components')
sources = nmf.fit(spectra)

pl.figure(figsize=(5, 10))
pl.subplots_adjust
for i in range(5):
    pl.subplot(511 + i)
    pl.plot(lam, nmf.components_[i])
    ylim = pl.ylim()
    pl.ylim(-0.1 * ylim[1], ylim[1])
pl.show()
