"""
===============================
Example of Plotting a Histogram
===============================

This example shows how to plot a histogram using matplotlib.
"""

import numpy as np
import pylab as pl

x1 = np.random.normal(0, 1, size=50000)
x2 = 8 * np.random.random(5000) ** 0.1 - 5

bins = np.linspace(-4, 4, 100)

pl.hist(np.concatenate([x1, x2]), bins, histtype='step')
pl.hist(x2, bins, histtype='step')

x = np.linspace(-4, 4, 1000)
y = (bins[1] - bins[0]) * 50000 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x ** 2)
pl.plot(x, y, '--k')

pl.xlabel('x label')
pl.ylabel('y label')
pl.title('Title')

pl.text(0.05, 0.9, 'Text goes here',
        fontsize=14, transform=pl.gca().transAxes,
        ha='left', va='bottom')

pl.show()
