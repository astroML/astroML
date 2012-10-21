"""
Examples of Plotting with Matplotlib
------------------------------------

This file generates the example plots from the Appendix
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)

#------------------------------------------------------------
# First Example: simple plot
plt.figure(1)
x = np.linspace(0, 2 * np.pi, 1000)
y = np.sin(x)
plt.plot(x, y)

plt.xlim(0, 2 * np.pi)
plt.ylim(-1.3, 1.3)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple Sinusoid Plot')

#------------------------------------------------------------
# Second Example: error-bars over simple plot
plt.figure(2)
x = np.linspace(0, 2 * np.pi, 1000)
y = np.sin(x)
plt.plot(x, y)

plt.xlim(0, 2 * np.pi)
plt.ylim(-1.3, 1.3)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple Sinusoid Plot')

x_obs = 2 * np.pi * np.random.random(50)
y_obs = np.sin(x_obs)
y_obs += np.random.normal(0, 0.1, 50)
plt.errorbar(x_obs, y_obs, 0.1, fmt='.', color='black')

#------------------------------------------------------------
# Third Example: histogram
plt.figure(3)
x = np.random.normal(size=1000)
plt.hist(x, bins=50)
plt.xlabel('x')
plt.ylabel('N(x)')

#------------------------------------------------------------
# Fourth Example: spline fitting
from scipy import interpolate

x = np.linspace(0, 16, 30)
y = np.sin(x)

x2 = np.linspace(0, 16, 1000)
spl = interpolate.UnivariateSpline(x, y, s=0)

plt.figure()
plt.plot(x, y, 'ok')
plt.plot(x2, spl(x2), '-k')
plt.ylim(-1.3, 1.3)

plt.show()
