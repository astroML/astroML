"""
=============================
Example of Plotting Errorbars
=============================

This example shows how to plot points with error bars using matplotlib.
"""

import numpy as np
import pylab as pl

norm = lambda x, x0, sig: (1. / np.sqrt(2 * np.pi * sig ** 2)
                           * np.exp(-0.5 * (x - x0) ** 2 / sig ** 2))

x = np.linspace(-6, 6, 1000)
y1 = 0.3 * norm(x, -3, 1.5)
y2 = norm(x, 0.5, 1)

y = y1 + y2

# select every 25 points as data
xdata = x[::25]
ydata = y[::25]
yerr = np.random.normal(0, 0.03, size=ydata.size)

pl.plot(x, y, '-', c='gray')
pl.plot(x, y1, ':', c='gray')
pl.plot(x, y2, ':', c='gray')
pl.errorbar(xdata, (ydata + yerr), 0.03, fmt='.k')

pl.xlabel('x label')
pl.ylabel('y label')
pl.title('Title')

pl.text(0.05, 0.9, 'Text goes here',
        fontsize=14, transform=pl.gca().transAxes,
        ha='left', va='bottom')

pl.show()
