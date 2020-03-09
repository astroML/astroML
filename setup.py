from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info


DESCRIPTION = "tools for machine learning and data mining in Astronomy"
LONG_DESCRIPTION = open('README.rst').read()
NAME = "astroML"
AUTHOR = "Jake VanderPlas"
AUTHOR_EMAIL = "vanderplas@astro.washington.edu"
MAINTAINER = "Brigitta Sipocz"
MAINTAINER_EMAIL = "brigitta.sipocz@gmail.com"
URL = 'http://astroML.github.com'
DOWNLOAD_URL = 'https://github.com/astroML/astroML'
LICENSE = 'BSD'

import astroML
VERSION = astroML.__version__

install_requires = ['scikit-learn>=0.18',
                    'numpy>=1.13',
                    'scipy>=0.18',
                    'matplotlib>=3.0',
                    'astropy>=3.0']


def trigger_theano():
    try:
        import pymc3 as pm
        print("Run small regression example to trigger theano builds, if pymc3 is available.")
        import numpy as np
        from astroML.linear_model import LinearRegressionwithErrors
        lr = LinearRegressionwithErrors()
        x = np.arange(10)
        y = np.arange(10) * 2 + 1
        x_err = np.ones(10) * 0.1
        y_err = np.ones(10) * 0.1
        lr.fit(x, y, y_err, x_err)
    except (ImportError, RuntimeError):
        pass


class CustomDevelopCommand(develop):
    """Customized setuptools develop command to trigger theano compilation."""
    def run(self):
        develop.run(self)
        trigger_theano()


class CustomEggInfoCommand(egg_info):
    """Customized setuptools egg_info command to trigger theano compilation."""
    def run(self):
        egg_info.run(self)
        trigger_theano()


class CustomInstallCommand(install):
    """Customized setuptools install command to trigger theano compilation."""
    def run(self):
        install.run(self)
        trigger_theano()


setup(cmdclass={'install': CustomInstallCommand,
                'develop': CustomDevelopCommand,
                'egg_info': CustomEggInfoCommand},
      name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      install_requires=install_requires,
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
                'astroML.classification.tests',
                'astroML.linear_model',
                'astroML.linear_model.tests',
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
                'astroML.utils',
                'astroML.utils.tests',
            ],
      python_requires='>=3.5',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Astronomy'],
     )
