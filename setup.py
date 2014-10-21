from distutils.core import setup

DESCRIPTION = "tools for machine learning and data mining in Astronomy"
LONG_DESCRIPTION = open('README.rst').read()
NAME = "astroML"
AUTHOR = "Jake VanderPlas"
AUTHOR_EMAIL = "vanderplas@astro.washington.edu"
MAINTAINER = "Jake VanderPlas"
MAINTAINER_EMAIL = "vanderplas@astro.washington.edu"
URL = 'http://astroML.github.com'
DOWNLOAD_URL = 'http://github.com/astroML/astroML'
LICENSE = 'BSD'

import astroML
VERSION = astroML.__version__

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=['astroML',
                'astroML.tests',
                'astroML.clustering',
                'astroML.clustering.tests',
                'astroML.classification',
                'astroML.linear_model',
                'astroML.datasets',
                'astroML.datasets.tools',
                'astroML.density_estimation',
                'astroML.density_estimation.tests',
                'astroML.time_series',
                'astroML.time_series.tests',
                'astroML.dimensionality',
                'astroML.dimensionality.tests',
                'astroML.plotting',
                'astroML.plotting.tests',
                'astroML.stats',
                'astroML.stats.tests',
            ],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Astronomy'],
     )
