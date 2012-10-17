"""
====================
Scatter Plot Example
====================

This example shows how to create a scatter plot using matplotlib.
"""

import numpy as np
import pylab as pl

x = np.random.random(10000)
y = 1. / (x + 0.1)
c = x + np.random.normal(0, 0.1, size=x.size)

x += np.random.normal(0, 0.1, size=x.size)
y += np.random.normal(0, 0.1, size=y.size)

pl.scatter(x, y, c=c, s=9, lw=0)
cb = pl.colorbar()

cb.set_label('color label')
pl.xlabel('x label')
pl.ylabel('y label')
pl.title('Title')
pl.text(0.95, 0.9, r'$\mathdefault{y \approx (x + 0.1)^{-1}}$',
        fontsize=14,
        ha='right', va='bottom', transform=pl.gca().transAxes)
pl.show()
