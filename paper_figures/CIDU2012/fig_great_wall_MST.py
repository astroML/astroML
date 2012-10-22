"""
Euclidean Minimum Spanning Tree
-------------------------------
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure is an example from astroML: see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt

from scipy.interpolate import interp1d
from sklearn.neighbors import kneighbors_graph

try:
    from scipy.sparse.csgraph import minimum_spanning_tree
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
# Get the x, y coordinates of the beginning and end of each line segment
T = T.tocoo()

dist = T.data
p1 = T.row
p2 = T.col

A = X[p1].T
B = X[p2].T

x_coords = np.vstack([A[0], B[0]])
y_coords = np.vstack([A[1], B[1]])

#----------------------------------------------------------------------
# Plot the results
fig = plt.figure()
fig.subplots_adjust(hspace=0, left=0.1, right=0.95, bottom=0.1, top=0.9)

ax = fig.add_subplot(211, aspect='equal')
ax.scatter(X[:, 1], X[:, 0], s=1, lw=0, c='k')
ax.set_xlim(ymin, ymax)
ax.set_ylim(xmin, xmax)
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.set_ylabel('x (Mpc)')

ax = fig.add_subplot(212, aspect='equal')
ax.plot(y_coords, x_coords, c='k', lw=1)
ax.set_xlim(ymin, ymax)
ax.set_ylim(xmin, xmax)
ax.set_xlabel('y (Mpc)')
ax.set_ylabel('x (Mpc)')

plt.show()
