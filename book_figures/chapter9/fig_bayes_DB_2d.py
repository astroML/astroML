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

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=True)

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
fig = plt.figure(figsize=(5, 5), facecolor='w')
ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])

# draw axes
plt.annotate(r'$x_1$', (-0.08, -0.02), (1.05, -0.02),
             ha='center', va='center',
             arrowprops=dict(arrowstyle='<-', color='k'))
plt.annotate(r'$x_2$', (-0.02, -0.08), (-0.02, 1.05),
             ha='center', va='center',
             arrowprops=dict(arrowstyle='<-', color='k'))

# draw ellipses, points, and boundaries
ax.scatter(mu1[:1], mu1[1:], c='k')
ax.scatter(mu2[:1], mu2[1:], c='k')

ax.add_patch(Ellipse(mu1, sigma1[0], sigma1[1], fc='none', ec='k'))
ax.add_patch(Ellipse(mu2, sigma2[0], sigma2[1], fc='none', ec='k'))

ax.text(mu1[0] + 0.02, mu1[1] + 0.02, r'$\mu_1$')
ax.text(mu2[0] + 0.02, mu2[1] + 0.02, r'$\mu_2$')

ax.plot(x_boundary, y_boundary, '--k')
ax.text(0.53, 0.28, "decision boundary", rotation=-70,
        ha='left', va='bottom')

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)

plt.show()
