"""
Example of a Discriminant Function
----------------------------------
This plot shows a simple example of a discriminant function between
two sets of points
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt

#------------------------------------------------------------
# create some toy data
np.random.seed(0)
cluster_1 = np.random.normal([1, 0.5], 0.5, size=(10, 2))
cluster_2 = np.random.normal([-1, -0.5], 0.5, size=(10, 2))

#------------------------------------------------------------
# plot the data and boundary
fig = plt.figure()
ax = fig.add_subplot(111, xticks=[], yticks=[])
ax.scatter(cluster_1[:, 0], cluster_1[:, 1], c='k', s=50)
ax.scatter(cluster_2[:, 0], cluster_2[:, 1], c='w', s=50)
ax.plot([0, 1], [1.5, -1.5], '-k', lw=2)

ax.set_xlim(-2, 2.5)
ax.set_ylim(-2, 2)

plt.show()
