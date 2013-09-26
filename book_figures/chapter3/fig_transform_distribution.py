r"""
Transformation of Distribution
------------------------------
Figure 3.4.

An example of transforming a uniform distribution. In the left panel, x
is sampled from a uniform distribution of unit width centered on x = 0.5
(:math:`\mu` = 0 and W = 1; see Section 3.3.1). In the right panel,
the distribution is transformed via y = exp(x). The form of the resulting
pdf is computed from eq. 3.20.
"""
# Author: Jake VanderPlas
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=True)

#------------------------------------------------------------
# Set up the data
np.random.seed(0)

# create a uniform distribution
uniform_dist = stats.uniform(0, 1)
x_sample = uniform_dist.rvs(1000)
x = np.linspace(-0.5, 1.5, 1000)
Px = uniform_dist.pdf(x)

# transform the data
y_sample = np.exp(x_sample)
y = np.exp(x)
Py = Px / y

#------------------------------------------------------------
# Plot the results
fig = plt.figure(figsize=(5, 2.5))
fig.subplots_adjust(left=0.11, right=0.95, wspace=0.3, bottom=0.17, top=0.9)

ax = fig.add_subplot(121)
ax.hist(x_sample, 20, histtype='stepfilled', fc='#CCCCCC', normed=True)
ax.plot(x, Px, '-k')
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(0, 1.4001)
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
ax.text(0.95, 0.95, r'$p_x(x) = {\rm Uniform}(x)$',
        va='top', ha='right',
        transform=ax.transAxes)
ax.set_xlabel('$x$')
ax.set_ylabel('$p_x(x)$')


ax = fig.add_subplot(122)
ax.hist(y_sample, 20, histtype='stepfilled', fc='#CCCCCC', normed=True)
ax.plot(y, Py, '-k')
ax.set_xlim(0.85, 2.9)
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
ax.text(0.95, 0.95, '$y=\exp(x)$\n$p_y(y)=p_x(\ln y) / y$',
        va='top', ha='right',
        transform=ax.transAxes)
ax.set_xlabel('$y$')
ax.set_ylabel('$p_y(y)$')

plt.show()
