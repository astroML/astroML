"""
================================
SDSS Spectroscopic Galaxy Sample
================================
"""
import numpy as np
import pylab as pl

from astroML.datasets import fetch_sdss_specgals

data = fetch_sdss_specgals()

#------------------------------------------------------------
# plot the RA/DEC in an area-preserving projection

RA = data['ra']
DEC = data['dec']

# convert coordinates to degrees
RA -= 180
RA *= np.pi / 180
DEC *= np.pi / 180

ax = pl.axes(projection='mollweide')

ax = pl.axes()
pl.grid(True)
pl.scatter(RA, DEC, s=1, lw=0, c=data['z'], cmap=pl.cm.copper,
           vmin=0, vmax=0.4)

pl.title('SDSS DR8 Spectroscopic Galaxies')
cb = pl.colorbar(cax = pl.axes([0.05, 0.1, 0.9, 0.05]),
                 orientation='horizontal',
                 ticks=np.linspace(0, 0.4, 9))
cb.set_label('redshift')


#------------------------------------------------------------
# plot the r vs u-r color-magnitude diagram
u = data['modelMag_u']
r = data['modelMag_r']
rPetro = data['petroMag_r']

pl.figure()
ax = pl.axes()
pl.scatter(u - r, rPetro, s=1, lw=0, c=data['z'], cmap=pl.cm.copper,
           vmin=0, vmax=0.4)
pl.colorbar(ticks=np.linspace(0, 0.4, 9)).set_label('redshift')

pl.xlim(0.5, 5.5)
pl.ylim(18, 12.5)

pl.xlabel('u-r')
pl.ylabel('rPetrosian')

#------------------------------------------------------------
# plot a histogram of the redshift

from astroML.density_estimation import knuth_nbins

pl.figure()
pl.hist(data['z'], knuth_nbins(data['z']),
        histtype='stepfilled', ec='k', fc='#F5CCB0')
pl.xlim(0, 0.4)
pl.xlabel('z (redshift)')
pl.ylabel('dN/dz(z)')

pl.show()
