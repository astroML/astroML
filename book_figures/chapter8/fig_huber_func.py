"""
Huber Functional Form
---------------------
This plots the form of the huber loss function :math:`\Phi(t)`.
"""
# Author: Jake VanderPlas
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general
import numpy as np
from matplotlib import pyplot as plt

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=True)


#------------------------------------------------------------
# Define the Huber loss
def Phi(t, c):
    t = abs(t)
    flag = (t > c)
    return (~flag) * (0.5 * t ** 2) - (flag) * c * (0.5 * c - t)

#------------------------------------------------------------
# Plot for several values of c
fig = plt.figure(figsize=(5, 3.75))
ax = fig.add_subplot(111)

x = np.linspace(-10, 10, 100)

for c in (1, 2, 3, 5, 1000):
    y = Phi(x, c)
    ax.plot(x, y, '-k')

    if c > 10:
        s = r'\infty'
    else:
        s = str(c)

    ax.text(x[6], y[6], '$c=%s$' % s,
            ha='center', va='center',
            bbox=dict(boxstyle='round', ec='k', fc='w'))

ax.set_xlabel('$t$')
ax.set_ylabel(r'$\Phi(t)$')

plt.show()
