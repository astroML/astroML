"""
Coin Odds Ratio
---------------

This figure shows the odds ratio for a coin flip.  The curves show the
odds ratio between model :math:`M_1`, in which the probability of landing
heads is known to be :math:`b^*`, and model :math:`M_2`, where the probability
of landing heads is unknown.

Here we plot the odds ratio between the models, :math:`O_{21}`, as a function
of :math:`k` heads observed :math:`n` coin tosses.  Comparing the panels, it
is clear that, as expected, observing more tosses gives better constraints
on models via the odds ratio.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt


@np.vectorize
def odds_ratio(n, k, bstar):
    """Odds ratio between M_2, where the heads probability is unknown,
    and M_1, where the heads probability is known to be `bstar`, evaluated
    in the case of `k` heads observed in `n` tosses.

    Eqn. 5.25 in the text
    """
    factor = 1. / (bstar ** k * (1 - bstar) ** (n - k))
    f = lambda b: b ** k * (1 - b) ** (n - k)

    return factor * integrate.quad(f, 0, 1)[0]

#------------------------------------------------------------
# Plot the results
fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(left=0.1, right=0.95, wspace=0.05, bottom=0.12)

subplots = [121, 122]
n_array = [10, 20]

linestyles = ['-g', '--b']
bstar_array = [0.5, 0.1]

for subplot, n in zip(subplots, n_array):
    ax = fig.add_subplot(subplot, yscale='log')
    k = np.arange(n + 1)

    # plot curves for two values of bstar
    for ls, bstar in zip(linestyles, bstar_array):
        ax.plot(k, odds_ratio(n, k, bstar), ls,
                label=r'$b^* = %.1f$' % bstar)

    if subplot == 121:
        ax.set_ylabel(r'$O_{21}$')
        ax.legend(loc=2)
    else:
        ax.set_xlim(0.01, n)
        ax.yaxis.set_major_formatter(plt.NullFormatter())

    ax.set_xlabel('k')
    ax.set_title('n = %i' % n)
    ax.set_ylim(1E-1, 1E4)
    ax.xaxis.set_major_locator(plt.MultipleLocator(n / 5))
    ax.grid()


plt.show()
