"""
2D Bayes Decision Boundary
--------------------------
Plot a schematic of a two-dimensional decision boundary
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

#------------------------------------------------------------
# Set up diagram
mu1 = (0.25, 0.25)
mu2 = (0.85, 0.7)

sigma1 = (0.5, 0.5)
sigma2 = (0.25, 0.5)

y_boundary = np.linspace(-0.1, 1.1, 100)
x_boundary = (0.5 + 0.4 * (y_boundary - 0.9) ** 2)

#------------------------------------------------------------
# Set up plot
fig = plt.figure(figsize=(6, 6), facecolor='w')
ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])

# draw axes
plt.annotate(r'$x_1$', (-0.08, -0.02), (1.05, -0.02),
             ha='center', va='center', fontsize=16,
             arrowprops=dict(arrowstyle='<-', color='k', lw=1.5))
plt.annotate(r'$x_2$', (-0.02, -0.08), (-0.02, 1.05),
             ha='center', va='center', fontsize=16,
             arrowprops=dict(arrowstyle='<-', color='k', lw=1.5))

# draw ellipses, points, and boundaries
ax.scatter(mu1[:1], mu1[1:], c='k')
ax.scatter(mu2[:1], mu2[1:], c='k')

ax.add_patch(Ellipse(mu1, sigma1[0], sigma1[1], fc='none', ec='k', lw=2))
ax.add_patch(Ellipse(mu2, sigma2[0], sigma2[1], fc='none', ec='k', lw=2))

ax.text(mu1[0] + 0.02, mu1[1] + 0.02, r'$\mu_1$', size=18)
ax.text(mu2[0] + 0.02, mu2[1] + 0.02, r'$\mu_2$', size=18)

ax.plot(x_boundary, y_boundary, '--k', lw=2)
ax.text(0.53, 0.28, "decision boundary", rotation=-70, fontsize=16,
        ha='left', va='bottom')

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)

plt.show()
