"""
Lynden-Bell C- setup
--------------------
This diagram shows a schematic of the Lynden-Bell C- method.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

#------------------------------------------------------------
# Draw the schematic
fig = plt.figure(figsize=(8, 4))
fig.subplots_adjust(left=0.05, right=0.95, wspace=0.1)
ax1 = fig.add_subplot(121, xticks=[], yticks=[])
ax2 = fig.add_subplot(122, xticks=[], yticks=[])

# define a convenient function
max_func = lambda t: 1. / (0.5 + t) - 0.5

x = np.linspace(0, 1.0, 100)
ymax = max_func(x)
ymax[ymax > 1] = 1

# draw and label the common background
for ax in (ax1, ax2):
    ax.fill_between(x, 0, ymax, color='gray', alpha=0.5)

    ax.plot([-0.1, 1], [1, 1], '--k', lw=1)

    ax.text(0.7, 0.35, '$y_{max}(x)$', rotation=-30)

    ax.plot([1, 1], [0, 1], '--k', lw=1)
    ax.text(1.01, 0.5, '$x_{max}$', ha='left', va='center', rotation=90)

# draw and label J_i in the first axes
xi = 0.4
yi = 0.35
ax1.scatter([xi], [yi], s=16, lw=0, c='k')
ax1.text(xi, yi, ' $(x_i, y_i)$', ha='left', va='center')
ax1.add_patch(Rectangle((0, 0), xi, max_func(xi), ec='k', fc='gray',
                        linestyle='dashed', lw=1, alpha=0.5))
ax1.text(0.5 * xi, 0.5 * max_func(xi), '$J_i$', ha='center', va='center')

# draw and label J_k in the second axes
ax2.scatter([xi], [yi], s=16, lw=0, c='k')
ax2.text(xi, yi, ' $(x_k, y_k)$', ha='center', va='bottom')
ax2.add_patch(Rectangle((0, 0), max_func(yi), yi, ec='k', fc='gray',
                        linestyle='dashed', lw=1, alpha=0.5))
ax2.text(0.5 * max_func(yi), 0.5 * yi, '$J_k$', ha='center', va='center')

# adjust the limits of both axes
for ax in (ax1, ax2):
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

plt.show()
