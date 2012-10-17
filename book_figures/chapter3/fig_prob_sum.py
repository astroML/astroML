"""
Sum of Probabilities
--------------------
Diagram of a sum of probabilities
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning for Astronomy" (2013)
#   For more information, see http://astroML.github.com
import pylab as pl

# create plot
fig = pl.figure(figsize=(8, 6), facecolor='w')
ax = pl.axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)

# draw intersecting circles
ax.add_patch(pl.Circle((1.5, 0.2), 2.2, fc='gray', ec='black', alpha=0.5))
ax.add_patch(pl.Circle((-1.5, 0.2), 2.2, fc='gray', ec='black', alpha=0.5))

# add text
text_kwargs = dict(ha='center', va='center', fontsize=20)
ax.text(-1.6, 0.2, "$p(A)$", **text_kwargs)
ax.text(1.6, 0.2, "$p(B)$", **text_kwargs)
ax.text(0.0, 0.2, "$p(A \cap B)$", **text_kwargs)
ax.text(0, -2.3, "$p(A \cup B) = p(A) + p(B) - p(A \cap B)$", **text_kwargs)

ax.set_xlim(-4, 4)
ax.set_ylim(-3, 3)

pl.show()
