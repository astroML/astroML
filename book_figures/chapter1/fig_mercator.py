"""
Mercator Projection
-------------------
Figure1.13.

The Mercator projection. Shown are the projections of circles of constant
radius 10 degrees across the sky. Note that the area is not preserved by the
Mercator projection: the projection increases the size of finite regions on
the sphere, with a magnitude which increases at high latitudes.
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
from astroML.plotting import plot_tissot_ellipse

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=True)


#------------------------------------------------------------
# generate a latitude/longitude grid
circ_long = np.linspace(-np.pi, np.pi, 13)[1:-1]
circ_lat = np.linspace(-np.pi / 2, np.pi / 2, 7)[1:-1]
radius = 10 * np.pi / 180.



#------------------------------------------------------------
# plot Mercator projection: we need to set this up manually
def mercator_axes():
    ax = plt.axes(aspect=1.0)
    ax.set_xticks(np.pi / 6 * np.linspace(-5, 5, 11))
    ax.set_yticks(np.pi / 12 * np.linspace(-5, 5, 11))
    for axy in (ax.xaxis, ax.yaxis):
        axy.set_major_formatter(plt.FuncFormatter(lambda s, a: r'$%i^\circ$'
                                                  % np.round(s * 180 / np.pi)))
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi / 2, np.pi / 2)

    return ax

plt.figure(figsize=(5, 3.75))
ax = mercator_axes()
ax.grid(True)
plot_tissot_ellipse(circ_long[:, None], circ_lat, radius,
                    ax=ax, fc='k', alpha=0.3, lw=0)
ax.set_title('Mercator projection')

plt.show()
