"""
Lasso Regression Diagram
------------------------
This displays a schematic diagram of lasso regression
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, RegularPolygon

#------------------------------------------------------------
# set up plot
fig = plt.figure(figsize=(8, 8), facecolor='w')
ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])

# plot axes
ax.arrow(-1, 0, 9, 0, head_width=0.1, fc='k')
ax.arrow(0, -1, 0, 9, head_width=0.1, fc='k')

# plot ellipses and circles
for i in range(3):
    ax.add_patch(Ellipse((3, 5),
                         3.5 * np.sqrt(2 * i + 1), 1.7 * np.sqrt(2 * i + 1),
                         -15, fc='none', lw=2))

ax.add_patch(RegularPolygon((0, 0), 4, 4.4, np.pi, fc='none', lw=2))

# plot arrows
ax.arrow(0, 0, 0, 4.4, head_width=0.2, fc='k', length_includes_head=True)
ax.arrow(0, 0, 3, 5, head_width=0.2, fc='k', length_includes_head=True)
ax.arrow(0, -0.2, 4.2, 0, head_width=0.1, fc='k', length_includes_head=True)
ax.arrow(4.2, -0.2, -4.2, 0, head_width=0.1, fc='k', length_includes_head=True)

# annotate plot
ax.text(7.5, -0.1, r'$\theta_1$', va='top', fontsize=20)
ax.text(-0.1, 7.5, r'$\theta_2$', ha='right', fontsize=20)
ax.text(3, 5 + 0.2, r'$\rm \theta_{normal\ equation}$', fontsize=20,
        ha='center', bbox=dict(boxstyle='round', ec='k', fc='w', alpha=0.7))
ax.text(0, 4.4 + 0.2, r'$\rm \theta_{lasso}$', fontsize=20, ha='center',
        bbox=dict(boxstyle='round', ec='k', fc='w', alpha=0.7))
ax.text(2, -0.3, r'$r$', fontsize=20, ha='center', va='top')

ax.set_xlim(-2, 9)
ax.set_ylim(-2, 9)

plt.show()
