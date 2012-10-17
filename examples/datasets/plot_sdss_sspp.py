"""
===========================================
SDSS SEGUE Stellar Parameters Pipeline Data
===========================================
"""

import numpy as np
import pylab as pl

from astroML.datasets import fetch_sdss_sspp
from astroML.stats import binned_statistic_2d

data = fetch_sdss_sspp()

# do some reasonable magnitude cuts
rpsf = data['rpsf']
data = data[(rpsf > 15) & (rpsf < 19)]

# get the desired data
logg = data['logg']
Teff = data['Teff']
alphFe = data['alphFe']
FeH = data['FeH']

statistic='mean'
bins=100

N, xedges, yedges = binned_statistic_2d(Teff, logg, FeH,
                                        'count', bins=bins)
alphFe_mean, xedges, yedges = binned_statistic_2d(Teff, logg, alphFe,
                                                  statistic, bins=bins)
FeH_mean, xedges, yedges = binned_statistic_2d(Teff, logg, FeH,
                                               statistic, bins=bins)

# Set pixels with no sources to white
cmap = pl.cm.jet
cmap.set_bad('w', 1.)

# Show the results
pl.figure()
pl.imshow(alphFe_mean.T, origin='lower',
          extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]],
          aspect='auto', interpolation='nearest', cmap=cmap)
pl.xlim(xedges[-1], xedges[0])
pl.ylim(yedges[-1], yedges[0])
pl.xlabel(r'$\mathrm{T_{eff}}$')
pl.ylabel(r'$\mathrm{log(g)}$')
pl.colorbar().set_label(r'$\mathrm{%s\ [\alpha/Fe]\ in\ pixel}$' % statistic)
pl.title('SEGUE Stellar Metallicity')


pl.figure()
pl.imshow(FeH_mean.T, origin='lower',
          extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]],
          aspect='auto', interpolation='nearest', cmap=cmap)
pl.xlim(xedges[-1], xedges[0])
pl.ylim(yedges[-1], yedges[0])
pl.xlabel(r'$\mathrm{T_{eff}}$')
pl.ylabel(r'$\mathrm{log(g)}$')
pl.colorbar().set_label(r'$\mathrm{%s\ [Fe/H]\ in\ pixel}$' % statistic)
pl.title('SEGUE Stellar Metallicity')


pl.figure()
pl.imshow(np.log10(N.T), origin='lower',
          extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]],
          aspect='auto', interpolation='nearest', cmap=cmap)
pl.xlim(xedges[-1], xedges[0])
pl.ylim(yedges[-1], yedges[0])
pl.xlabel(r'$\mathrm{T_{eff}}$')
pl.ylabel(r'$\mathrm{log(g)}$')
cb = pl.colorbar(ticks=[0, 1, 2, 3],
                 format=r'$10^{%i}$')
cb.set_label(r'$\mathrm{number\ in\ pixel}$')
pl.clim(0, 3)
pl.title('SDSS SEGUE Stellar Parameters')
pl.show()
