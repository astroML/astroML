"""
SDSS Fits file interface
------------------------

This demonstrates how to automatically download a spectrum from the SDSS
Data Archive Server, and to create a simple plot the resulting spectrum.
"""

from matplotlib import pyplot as plt
from astroML.datasets import fetch_sdss_spectrum

plate = 1615
mjd = 53166
fiber = 513

spec = fetch_sdss_spectrum(plate, mjd, fiber)

plt.plot(spec.wavelength(), spec.spectrum, '-k')
plt.xlabel(r'$\lambda (\AA)$')
plt.ylabel('Flux')
plt.title('Plate = %(plate)i, MJD = %(mjd)i, Fiber = %(fiber)i' % locals())
plt.show()
