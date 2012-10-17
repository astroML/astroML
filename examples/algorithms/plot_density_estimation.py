"""
Density Estimation
------------------
This plot compares k-Nearest Neighbors (kNN) density estimation to
Kernel Denstiy Estimation (KDE).
"""

import numpy as np
import pylab as pl
from scipy import stats

from astroML.algorithms import knn_local_density, kernel_density_estimator
from astroML.datasets import fetch_sdss_sspp

np.random.seed(0)

N1x = stats.norm(0, 0.2)
N1y = stats.norm(1, 0.2)

N2x = stats.norm(1, 0.4)
N2y = stats.norm(0, 0.2)

# compute the density as a function of x and y
def density(x, y, Npts):
    return Npts * (2. / 9. +
                   N1x.pdf(x) * N1y.pdf(y)
                   + N2x.pdf(x) * N2y.pdf(y)) / 4.

# draw a random sample from the above density
def rand(N):
    x1 = N1x.rvs(N / 4)
    y1 = N1y.rvs(N / 4)

    x2 = N2x.rvs(N / 4)
    y2 = N2y.rvs(N / 4)

    x3 = -1 + 3 * np.random.random(N - N / 4 - N / 4)
    y3 = -1 + 3 * np.random.random(N - N / 4 - N / 4)

    x = np.concatenate([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])
    
    return x, y

x = np.linspace(-1, 2, 100)
y = np.linspace(-1, 2, 100)

dx = x[1] - x[0]
dy = y[1] - y[0]

Npts = 200

pl.figure(figsize=(6, 10))
pl.subplots_adjust(left=0.05, right=0.95,
                   bottom=0.03, top=0.95,
                   wspace=0.1, hspace=0.2)

density = density(x, y[:, np.newaxis], Npts)

levels = 50
clim = (np.log10(density.min()), np.log10(density.max()))

print "density contrast: %.2g" % (density.max() / density.min())

#------------------------------------------------------------
# plot the true density
pl.subplot(321, xticks=[], yticks=[])
pl.contourf(x, y, np.log10(density), levels)
pl.clim(clim)
pl.title(r'$\mathrm{True\ Density}$')

#------------------------------------------------------------
# plot a realization from the density
pl.subplot(322, xticks=[], yticks=[])
x, y = rand(Npts)
pl.scatter(x, y, c='k', lw=0, s=4)
pl.xlim(-1, 2)
pl.ylim(-1, 2)
pl.title(r'$\mathrm{%i\ random\ points}$' % Npts)

#------------------------------------------------------------
# plot kNN estimation
for i, n_neighbors in enumerate([5, 20]):
    pl.subplot(323 + i, xticks=[], yticks=[])
    density, edges = knn_local_density([x, y], [50, 50],
                                       range=[(-1, 2), (-1, 2)],
                                       n_neighbors=n_neighbors)

    xcontour = 0.5 * (edges[0][:-1] + edges[0][1:])
    ycontour = 0.5 * (edges[1][:-1] + edges[1][1:])
    pl.contourf(xcontour, ycontour, np.log10(density.T), levels)
    pl.clim(clim)
    pl.title(r'$\mathrm{kNN\ estimator\ (k=%i)}$' % n_neighbors)

#------------------------------------------------------------
# plot kernel density estimation
for i, sigma in enumerate([0.1, 0.2]):
    pl.subplot(325 + i, xticks=[], yticks=[])
    
    density, edges = kernel_density_estimator([x, y], [50, 50],
                                              range=[(-1, 2), (-1, 2)],
                                              sigma=sigma)

    xcontour = 0.5 * (edges[0][:-1] + edges[0][1:])
    ycontour = 0.5 * (edges[1][:-1] + edges[1][1:])
    pl.contourf(xcontour, ycontour, np.log10(density.T), levels)
    pl.clim(clim)
    pl.title(r'$\mathrm{Kernel\ estimator\ (\sigma=%i)}$' % (10 * sigma))

#============================================================
# plot density estimation on SSPP dataset


# fetch the data
data = fetch_sdss_sspp()
data = data[:20000]

# do some reasonable magnitude cuts
rpsf = data['rpsf']
data = data[(rpsf > 15) & (rpsf < 19)]

# get the desired data
logg = data['logg']
Teff = data['Teff']

density, edges = knn_local_density([Teff, logg], 50,
                                   n_neighbors=50,
                                   range=[(4500, 8000),
                                          (1.0, 5.0)],
                                   normalize_input_range=True)

x = 0.5 * (edges[0][1:] + edges[0][:-1])
y = 0.5 * (edges[1][1:] + edges[1][:-1])

pl.figure(figsize=(6, 10))
pl.subplot(211)
pl.scatter(Teff, logg, s=1, c='k', lw=0, cmap=pl.cm.bone)
pl.contour(x, y, np.log10(density.T), 10,
           colors='r', linewidths=1, linestyles='-')
pl.xlim(8000, 4500)
pl.ylim(5.0, 1.0)

pl.subplot(212)
pl.imshow(np.log10(density.T), origin='lower', aspect='auto',
          extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]],
          cmap=pl.cm.binary)

pl.xlim(8000, 4500)
pl.ylim(5.0, 1.0)

pl.show()
