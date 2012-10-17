"""
Neighbor Search Scaling
-----------------------
This plots the scaling with N of an (RA, DEC) cross-matching for brute force
and KD Tree methods.
"""

import os, sys
from time import time

import numpy as np
import pylab as pl

from astroML.datasets import fetch_imaging_sample
from sklearn.neighbors import NearestNeighbors

data = fetch_imaging_sample()

X = np.empty((len(data), 2), dtype=np.float64)
X[:, 0] = data['ra']
X[:, 1] = data['dec']

#============================================================
# Plot scaling of neighbor search with number

Nsteps = 10
algorithms = dict(brute=5000,
                  kd_tree=data.shape[0])

for alg, maxN in algorithms.iteritems():
    nbrs = NearestNeighbors(n_neighbors=1, algorithm=alg,
                            warn_on_equidistant=False)
    t_query = np.zeros(Nsteps)
    t_build = np.zeros(Nsteps)
    sample_size = np.logspace(1, np.log10(maxN), Nsteps).astype(int)

    for i in range(Nsteps):
        Xi = X[:sample_size[i]]

        # for faster execution times, random fluctuations can be a problem.
        # deal with this by taking the median of several runs.
        N_repetitions = 1 + int(np.log(maxN / sample_size[i]))
        L_build = np.zeros(N_repetitions)
        L_query = np.zeros(N_repetitions)
        for j in range(N_repetitions):
            t0 = time()
            nbrs.fit(Xi)
            t1 = time()
            nbrs.kneighbors(Xi)
            t2 = time()

            L_build[j] = t1 - t0
            L_query[j] = t2 - t1
        t_build[i] = np.median(L_build)
        t_query[i] = np.median(L_query)

    l = pl.loglog(sample_size, t_query, '-o',
                  label=alg + ' query')
    pl.plot(sample_size, t_build, '--', color=l[0].get_color(),
            label=alg + ' setup')

pl.legend(loc=2)
pl.grid()
pl.xlabel("Number of Objects")
pl.ylabel("computation time (seconds)")
pl.title('(RA, DEC) nearest neighbor search')


pl.show()
