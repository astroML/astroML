"""
SDSS Line-ratio Diagrams
========================
This shows how to plot line-ratio diagrams for the SDSS spectra.  The cutoffs
are used to choose labels for the SDSS spectra projections.  The location of
the dividing line is taken from Kewley et al 2001.
"""
import numpy as np
import pylab as pl

from astroML.datasets import fetch_sdss_corrected_spectra
from astroML.datasets.tools.sdss_fits import log_OIII_Hb_NII

data = fetch_sdss_corrected_spectra()

i = np.where((data['lineindex_cln'] == 4) | (data['lineindex_cln'] == 5))

pl.scatter(data['log_NII_Ha'][i], data['log_OIII_Hb'][i],
           c=data['lineindex_cln'][i], s=9, lw=0)

NII = np.linspace(-2.0, 0.35)
pl.plot(NII, log_OIII_Hb_NII(NII), '-k')
pl.plot(NII, log_OIII_Hb_NII(NII, 0.1), '--k')
pl.plot(NII, log_OIII_Hb_NII(NII, -0.1), '--k')
pl.xlim(-2.0, 1.0)
pl.ylim(-1.2, 1.5)

pl.xlabel(r'$\mathrm{log([NII]/H\alpha)}$', fontsize='large')
pl.ylabel(r'$\mathrm{log([OIII]/H\beta)}$', fontsize='large')
pl.show()
