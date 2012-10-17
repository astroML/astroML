from distutils.core import setup

import astroML
VERSION = astroML.__version__

setup(name='astroML',
      version=VERSION,
      description='Machine Learning for Astronomy',
      author='Jake VanderPlas',
      author_email='vanderplas@astro.washington.edu',
      url='http://astroML.github.com',
      license='bsd',
      packages=['astroML',
                'astroML.linear_model',
                'astroML.datasets', 'astroML.datasets.tools',
                'astroML.density_estimation',
                'astroML.time_series',
                'astroML.PCA',
                'astroML.plotting',
                'astroML.stats',
                'astroML']
     )
