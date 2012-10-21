"""
Bayes Decision Boundary
-----------------------
This figure plots a schematic of a decision boundary for classification
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

#------------------------------------------------------------
# Compute the two PDFs
x = np.linspace(-3, 7, 1000)
pdf1 = norm(0, 1).pdf(x)
pdf2 = norm(3, 1.5).pdf(x)
x_bound = x[np.where(pdf1 < pdf2)][0]

#------------------------------------------------------------
# Plot the pdfs and decision boundary
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, pdf1, '-k', lw=1)
ax.fill_between(x, pdf1, color='gray', alpha=0.5)

ax.plot(x, pdf2, '-k', lw=1)
ax.fill_between(x, pdf2, color='gray', alpha=0.5)

# plot decision boundary
ax.plot([x_bound, x_bound], [0, 0.5], '--k')

ax.text(x_bound + 0.2, 0.49, "decision boundary",
        ha='left', va='top', rotation=90)

ax.text(0, 0.2, '$g_1(x)$', ha='center', va='center',
        fontsize=18)
ax.text(3, 0.1, '$g_2(x)$', ha='center', va='center',
        fontsize=18)

ax.set_xlim(-2, 7)
ax.set_ylim(0, 0.5)

ax.set_xlabel('x')
ax.set_ylabel('p(x)')

plt.show()
