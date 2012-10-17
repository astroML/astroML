"""
SDSS spectra ICA
----------------

Plot the ICA compotnents of the SDSS spectra
"""
import numpy as np
import pylab as pl
from matplotlib import ticker

from sklearn.decomposition import FastICA

from astroML.plotting import multiscatter, multidensity
from astroML.datasets import\
    sdss_corrected_spectra, fetch_sdss_corrected_spectra

data = fetch_sdss_corrected_spectra()

spectra = data['spectra']
lam = sdss_corrected_spectra.compute_wavelengths(data)

spectra = spectra[:, :850]
lam = lam[:850]

ica = FastICA(5, algorithm='parallel')

# ICA is tricky here.  It assumes that observations happen in a sequence, so
# that each row is related to the next row.  We need to transpose the spectra,
# so that each "observation" is the set of fluxes at a given wavelength.

sources = ica.fit(spectra.T).transform(spectra.T)

pl.figure(figsize=(5, 10))
for i in range(5):
    pl.subplot(511 + i)
    pl.plot(lam, sources[:, i])
pl.show()



