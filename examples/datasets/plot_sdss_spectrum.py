"""
========================
SDSS Fits file interface
========================

This demonstrates how to automatically download a spectrum from the SDSS
Data Archive Server, and to create a simple plot the resulting spectrum.
"""

import pylab as pl
from astroML.datasets import fetch_sdss_spectrum

plate = 1615
mjd = 53166
fiber = 513

spec = fetch_sdss_spectrum(plate, mjd, fiber)

pl.plot(spec.wavelength(), spec.spectrum, '-k')
pl.xlabel(r'$\lambda (\AA)$')
pl.ylabel('Flux')
pl.title('Plate = %(plate)i, MJD = %(mjd)i, Fiber = %(fiber)i' % locals())
pl.show()
