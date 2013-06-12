"""
NASA Atlas dataset example
--------------------------
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

from astroML.datasets import fetch_nasa_atlas
from astroML.plotting.tools import devectorize_axes

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=True)

data = fetch_nasa_atlas()
RA = data['RA']
DEC = data['DEC']

# convert coordinates to degrees
RA -= 180
RA *= np.pi / 180
DEC *= np.pi / 180

fig = plt.figure(figsize=(5, 3.75))
ax = plt.axes(projection='mollweide')
plt.scatter(RA, DEC, s=1, lw=0, c=data['Z'], cmap=plt.cm.copper)
plt.grid(True)
# devectorize: otherwise the resulting pdf figure becomes very large
#devectorize_axes(ax, dpi=400)

plt.title('NASA Atlas Galaxy Locations')
cb = plt.colorbar(cax=plt.axes([0.05, 0.1, 0.9, 0.05]),
                  orientation='horizontal',
                  ticks=np.linspace(0, 0.05, 6))
cb.set_label('redshift')

plt.show()
