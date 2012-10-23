"""
Gaussian Kernel Expansion Diagram
---------------------------------
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com

import numpy as np
from matplotlib import pyplot as plt

plt.figure(facecolor='w')
ax = plt.axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])

ax.add_patch(plt.Rectangle((-0.5, -0.25), 0.8, 0.4,
                           fc='none', ec='k', lw=2))
ax.add_patch(plt.Rectangle((-1.75, 0.1), 0.8, 0.4,
                           fc='none', ec='k', lw=2, linestyle='dashed'))
ax.add_patch(plt.Rectangle((0.8, -0.55), 0.8, 0.4,
                           fc='none', ec='k', lw=2, linestyle='dashed'))
ax.add_patch(plt.Rectangle((-1.3, -0.95), 0.8, 0.4,
                           fc='none', ec='k', lw=2, linestyle='dashed'))

red_pts = np.array([[-0.163, 0.093],
                    [-0.123, -0.22],
                    [0.194, 0.035],
                    [0.146, -0.178],
                    [-0.387, -0.143]])

blue_pts = np.array([[-1.51, 0.17],
                     [-1.17, 0.36],
                     [-1.23, -0.68],
                     [-0.80, -0.83],
                     [1.28, -0.45],
                     [1.41, -0.26]])

x0 = -0.5 + 0.4
y0 = -0.25 + 0.2

ax.scatter(red_pts[:, 0], red_pts[:, 1], c='r')
ax.scatter(blue_pts[:, 0], blue_pts[:, 1], c='b')
ax.scatter([x0], [y0], c='gray')

for pt in blue_pts:
    ax.annotate(" ", pt, (x0, y0), arrowprops=dict(arrowstyle='->',
                                                   linestyle='dashed'))

for i, pt in enumerate(red_pts):
    ax.annotate(" ", pt, (x0, y0), arrowprops=dict(arrowstyle='<-'))
    ax.text(pt[0] + 0.03, pt[1] + 0.03, '$r_{j%i}$' % (i + 1),
            bbox=dict(boxstyle='round', ec='k', fc='w', alpha=0.7))

ax.annotate("R.c", (x0, y0), (0.2, 0.2),
            arrowprops=dict(arrowstyle='-', color='gray'),
            bbox=dict(boxstyle='round', ec='k', fc='w'))

ax.set_xlim(-1.9, 1.9)
ax.set_ylim(-1.2, 0.8)
plt.show()
