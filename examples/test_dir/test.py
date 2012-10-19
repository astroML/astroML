"""
Test plot
---------
This is my test plot
"""

import pylab as pl
import numpy as np

x = np.linspace(0, 10, 1000)

pl.plot(x, np.sin(x))
pl.show()
