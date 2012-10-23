"""
NASA Atlas dataset example
--------------------------
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt

from astroML.datasets import fetch_nasa_atlas
from astroML.plotting.tools import devectorize_axes

data = fetch_nasa_atlas()
RA = data['RA']
DEC = data['DEC']

# convert coordinates to degrees
RA -= 180
RA *= np.pi / 180
DEC *= np.pi / 180

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
