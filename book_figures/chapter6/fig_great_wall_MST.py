"""
Euclidean Minimum Spanning Tree
-------------------------------
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
import pylab as pl

from scipy import sparse
from sklearn.neighbors import kneighbors_graph
from sklearn.mixture import GMM

try:
    from scipy.sparse.csgraph import \
        minimum_spanning_tree, connected_components
except:
    raise ValueError("scipy v0.11 or greater required "
                     "for minimum spanning tree")

from astroML.datasets import fetch_great_wall
from astroML.cosmology import Cosmology

#------------------------------------------------------------
# get data
X = fetch_great_wall()

xmin, xmax = (-375, -175)
ymin, ymax = (-300, 200)

#------------------------------------------------------------
# generate a sparse graph using the k nearest neighbors of each point
G = kneighbors_graph(X, n_neighbors=10, mode='distance')

#------------------------------------------------------------
# Compute the minimum spanning tree of this graph
T = minimum_spanning_tree(G, overwrite=True)

#------------------------------------------------------------
# Compute clusters
edge_cutoff = 0.9
cluster_cutoff = 30

T_trunc = T.copy()

data = T_trunc.data
cutoff = np.sort(data)[int(edge_cutoff * len(data))]
T_trunc.data[T_trunc.data > cutoff] = 0
T_trunc.eliminate_zeros()

n_components, labels = connected_components(T_trunc, directed=False)


#------------------------------------------------------------
# Get the x, y coordinates of the beginning and end of each line segment
def get_graph_segments(T):
    T = T.tocoo()

    dist = T.data
    p1 = T.row
    p2 = T.col

    A = X[p1].T
    B = X[p2].T

    x_coords = np.vstack([A[0], B[0]])
    y_coords = np.vstack([A[1], B[1]])

    return x_coords, y_coords

T_x, T_y = get_graph_segments(T)

#------------------------------------------------------------
# Fit a GMM to each individual cluster
Nx = 100
Ny = 250
Xgrid = np.vstack(map(np.ravel, np.meshgrid(np.linspace(xmin, xmax, Nx),
                                            np.linspace(ymin, ymax, Ny)))).T
density = np.zeros(Xgrid.shape[0])

clusters = np.zeros(X.shape[0], dtype=bool)

for i in range(n_components):
    ind = (labels == i)
    if ind.sum() > cluster_cutoff:
        clusters |= ind
        gmm = GMM(4).fit(X[ind])
        dens = np.exp(gmm.score(Xgrid))
        dens /= dens.max()
        density += dens

density = density.reshape((Ny, Nx))

# eliminate links in T_trunc which are not clusters
I = sparse.eye(len(clusters), len(clusters))
I.data[0, ~clusters] = 0
T_trunc = I * T_trunc * I

Ttrunc_x, Ttrunc_y = get_graph_segments(T_trunc)


#----------------------------------------------------------------------
# Plot the results
fig = pl.figure(figsize=(7, 8))
fig.subplots_adjust(hspace=0, left=0.1, right=0.95, bottom=0.1, top=0.9)

ax = fig.add_subplot(311, aspect='equal')
ax.scatter(X[:, 1], X[:, 0], s=1, lw=0, c='k')
ax.set_xlim(ymin, ymax)
ax.set_ylim(xmin, xmax)
ax.xaxis.set_major_formatter(pl.NullFormatter())
ax.set_ylabel('x (Mpc)')

ax = fig.add_subplot(312, aspect='equal')
ax.plot(T_y, T_x, c='k', lw=1)
ax.set_xlim(ymin, ymax)
ax.set_ylim(xmin, xmax)
ax.xaxis.set_major_formatter(pl.NullFormatter())
ax.set_xlabel('y (Mpc)')
ax.set_ylabel('x (Mpc)')

ax = fig.add_subplot(313, aspect='equal')
ax.plot(Ttrunc_y, Ttrunc_x, c='k', lw=1)
#ax.scatter(X[clusters, 1], X[clusters, 0], c=labels[clusters], lw=0)

ax.imshow(density.T, origin='lower', cmap=pl.cm.binary,
          extent=[ymin, ymax, xmin, xmax])

ax.set_xlim(ymin, ymax)
ax.set_ylim(xmin, xmax)
ax.xaxis.set_major_formatter(pl.NullFormatter())
ax.set_xlabel('y (Mpc)')
ax.set_ylabel('x (Mpc)')

pl.show()
