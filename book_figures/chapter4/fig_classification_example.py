"""
Example of classification
--------------------------------

This figure shows a schematic of the boundary choice in a classification
problem, where sources S are being selected from backgrounds B.  This
particular choice is one that strives for completeness (no missed sources)
at the expense of contamination (misclassified background sources).
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

#------------------------------------------------------------
# Generate and draw the curves
x = np.linspace(50, 200, 1000)
p1 = 0.9 * norm(100, 10).pdf(x)
p2 = 0.1 * norm(150, 12).pdf(x)

ax = plt.axes()
ax.fill(x, p1, ec='k', fc='#AAAAAA', alpha=0.5)
ax.fill(x, p2, '-k', fc='#AAAAAA', alpha=0.5)

ax.plot([120, 120], [0.0, 0.04], '--k')

ax.text(100, 0.036, r'$h_B(x)$', ha='center', va='bottom', fontsize=18)
ax.text(150, 0.0035, r'$h_S(x)$', ha='center', va='bottom', fontsize=18)
ax.text(122, 0.0395, r'$x_c=120$', ha='left', va='top', fontsize=18)
ax.text(125, 0.01, r'$(x > x_c\ {\rm classified\ as\ sources})$', fontsize=18)

ax.set_xlim(50, 200)
ax.set_ylim(0, 0.04)

ax.set_xlabel('x')
ax.set_ylabel('p(x)')
plt.show()
