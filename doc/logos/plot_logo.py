"""
NASA Sloan Atlas
----------------

This shows some visualizations of the data from the NASA SDSS Atlas
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure is an example from astroML: see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

from astroML.datasets import fetch_nasa_atlas

data = fetch_nasa_atlas()

#------------------------------------------------------------
# plot the RA/DEC in an area-preserving projection

RA = data['RA']
DEC = data['DEC']

# convert coordinates to degrees
RA -= 180
RA *= np.pi / 180
DEC *= np.pi / 180

fig = plt.figure(figsize=(8, 2), facecolor='w')
ax = fig.add_axes([0.56, 0.1, 0.4, 0.8], projection='mollweide')
plt.scatter(RA, DEC, s=1, lw=0, c=data['Z'], cmap=plt.cm.copper)
plt.grid(True)
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.yaxis.set_major_formatter(plt.NullFormatter())

font = {'family' : 'neuropol X',
        'color'  : '#222222',
        'weight' : 'normal',
        'size'   : 135,
        }
fig.text(0.5, 0.5, 'astroML', ha='center', va='center',
         fontdict=font)
#size=135,
#fontproperties=FontProperties(['neuropol X bold', 'neuropol X']))

plt.savefig('logo.png')

plt.show()
