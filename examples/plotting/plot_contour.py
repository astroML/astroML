"""
=========================
Example of a Contour Plot
=========================

This example shows an example of a contour plot using matplotlib.
"""

import numpy as np
import pylab as pl

norm = lambda x, x0, sig: (1. / np.sqrt(2 * np.pi * sig ** 2)
                           * np.exp(-0.5 * (x - x0) ** 2 / sig ** 2))

x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)

z = norm(x, 0.5, 0.2) * norm(y[:, None], 0.5, 0.3)
zerr = np.random.normal(0, 0.2, size=z.shape)

pl.imshow(z + zerr, origin='lower', interpolation='nearest',
          extent=[x[0], x[-1], y[0], y[-1]])
cb = pl.colorbar()

pl.contour(x, y, z, colors='k', linewidths=2)

cb.set_label('colorbar label')

pl.xlabel('x label')
pl.ylabel('y label')
pl.title('Title')
pl.show()
