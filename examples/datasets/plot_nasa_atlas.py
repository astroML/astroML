"""
NASA Sloan Atlas
----------------

This shows some visualizations of the data from the NASA SDSS Atlas
"""
import numpy as np
import pylab as pl

from astroML.datasets import fetch_nasa_atlas

data = fetch_nasa_atlas()

#------------------------------------------------------------
# plot the RA/DEC in an area-preserving projection

RA = data['RA']
DEC = data['DEC']

# convert coordinates to degrees
RA -= 180
RA *= np.pi / 180
DEC *= np.pi / 180

ax = pl.axes(projection='mollweide')
pl.scatter(RA, DEC, s=1, lw=0, c=data['Z'], cmap=pl.cm.copper)
pl.grid(True)

pl.title('NASA Atlas Galaxy Locations')
cb = pl.colorbar(cax = pl.axes([0.05, 0.1, 0.9, 0.05]),
                 orientation='horizontal',
                 ticks=np.linspace(0, 0.05, 6))
cb.set_label('redshift')


#------------------------------------------------------------
# plot the r vs u-r color-magnitude diagram

absmag = data['ABSMAG']

u = absmag[:, 2]
r = absmag[:, 4]

pl.figure()
ax = pl.axes()
pl.scatter(u - r, r, s=1, lw=0, c=data['Z'], cmap=pl.cm.copper)
pl.colorbar(ticks=np.linspace(0, 0.05, 6)).set_label('redshift')

pl.xlim(0, 3.5)
pl.ylim(-10, -24)

pl.xlabel('u-r')
pl.ylabel('r')

#------------------------------------------------------------
# plot a histogram of the redshift
from astroML.plotting import hist

pl.figure()
hist(data['Z'], bins='knuth',
     histtype='stepfilled', ec='k', fc='#F5CCB0')
pl.xlabel('z')
pl.ylabel('N(z)')

pl.show()
