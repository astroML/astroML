"""
KD Dual-tree Diagram
----------------------
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)

#------------------------------------------------------------
ax = fig.add_subplot(211, xticks=[], yticks=[], aspect='equal')

x = np.array([[0.5],
              [0.5]])

Rx = np.array([[1.25, 1.30, 1.40, 1.52, 1.56],
               [0.50, 0.78, 0.22, 0.45, 0.64]])

ax.add_patch(plt.Rectangle((1.2, 0.2), 0.4, 0.6, fc='none', lw=2, zorder=2))

ax.scatter(x[0], x[1], c='r', s=30, zorder=2)
ax.scatter(Rx[0], Rx[1], c='b', s=30, zorder=2)

ax.arrow(0.5, 0.5, 0.7, 0, width=0.01, lw=0, color='gray',
         length_includes_head=True, zorder=1)
ax.arrow(0.5, 0.5, 1.1, -0.3, width=0.01, lw=0, color='gray',
         length_includes_head=True, zorder=1)

ax.text(x[0], x[1], r'$x_i$  ', ha='right', va='bottom', fontsize=18)
ax.text(1.6, 0.7, r' $R$', ha='left', va='bottom', fontsize=18)

ax.text(0.8, 0.5, r'$D^l(x_i, R)$', ha='left', va='bottom', fontsize=18)
ax.text(0.8, 0.25, r'$D^u(x_i, R)$', ha='left', va='bottom', fontsize=18)

ax.set_xlim(0.2, 1.8)
ax.set_ylim(0.1, 0.9)

#----------------------------------------------------------------------

ax = fig.add_subplot(212, xticks=[], yticks=[], aspect='equal')

Qx = Rx.copy()
Qx[0] -= 0.8
Qx[1] = 1.1 - Qx[1]

ax.add_patch(plt.Rectangle((0.4, 0.3), 0.4, 0.6, fc='none', lw=2, zorder=2))
ax.add_patch(plt.Rectangle((1.2, 0.2), 0.4, 0.6, fc='none', lw=2, zorder=2))
ax.scatter(Qx[0], Qx[1], c='r', s=30, zorder=2)
ax.scatter(Rx[0], Rx[1], c='b', s=30, zorder=2)

ax.arrow(0.8, 0.3, 0.4, 0, width=0.01, lw=0, color='gray',
         length_includes_head=True, zorder=1)
ax.arrow(0.4, 0.9, 1.2, -0.7, width=0.01, lw=0, color='gray',
         length_includes_head=True, zorder=1)

ax.text(0.4, 0.8, r'$Q$ ', ha='right', va='bottom', fontsize=18)
ax.text(1.6, 0.7, r' $R$', ha='left', va='bottom', fontsize=18)

ax.text(0.86, 0.65, r'$D^l(Q, R)$', ha='left', va='bottom', fontsize=18)
ax.text(0.86, 0.2, r'$D^u(Q, R)$', ha='left', va='bottom', fontsize=18)

ax.set_xlim(0.2, 1.8)
ax.set_ylim(0.15, 0.95)

plt.show()
