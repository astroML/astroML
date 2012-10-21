"""
Photometric Redshifts by Random Forests
---------------------------------------
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import itertools

import numpy as np
import pylab as pl

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from astroML.datasets import fetch_sdss_specgals

data = fetch_sdss_specgals()

# put magnitudes in a matrix
mag = np.vstack([data['modelMag_%s' % f] for f in 'ugriz']).T
z = data['z']

# train on ~60,000 points
mag_train = mag[::10]
z_train = z[::10]

# test on ~6,000 distinct points
mag_test = mag[1::100]
z_test = z[1::100]

def plot_results(z, z_fit, plotlabel=None,
                 xlabel=True, ylabel=True):
    pl.scatter(z[::1], z_fit[::1], s=1, lw=0, c='k')
    pl.plot([-0.1, 0.4], [-0.1, 0.4], ':k')
    pl.xlim(-0.02, 0.4001)
    pl.ylim(-0.02, 0.4001)
    pl.gca().xaxis.set_major_locator(pl.MultipleLocator(0.1))
    pl.gca().yaxis.set_major_locator(pl.MultipleLocator(0.1))
                
    if plotlabel:
        pl.text(0.03, 0.97, plotlabel,
                ha='left', va='top', transform=ax.transAxes)

    if xlabel:
        pl.xlabel(r'$\rm z_{true}$', fontsize=16)
    else:
        pl.gca().xaxis.set_major_formatter(pl.NullFormatter())
        
    if ylabel:
        pl.ylabel(r'$\rm z_{fit}$', fontsize=16)
    else:
        pl.gca().yaxis.set_major_formatter(pl.NullFormatter())
        

pl.figure(figsize=(8, 4))
pl.subplots_adjust(wspace=0.1,
                   left=0.1, right=0.95,
                   bottom=0.15, top=0.9)

ax = pl.subplot(121)
z_fit = DecisionTreeRegressor(max_depth=10).fit(mag_train,
                                                z_train).predict(mag_test)
print "one tree: rms =", np.sqrt(np.mean((z_test - z_fit) ** 2))
plot_results(z_test, z_fit, plotlabel="Decision Tree")

ax = pl.subplot(122)
z_fit = RandomForestRegressor(n_estimators=10,
                              max_depth=15).fit(mag_train,
                                                z_train).predict(mag_test)
print "ten trees: rms =", np.sqrt(np.mean((z_test - z_fit) ** 2))
plot_results(z_test, z_fit, plotlabel="Random Forest\nof 10 trees",
             ylabel=False)

pl.show()
