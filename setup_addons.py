import os
import numpy
from numpy.distutils.core import setup

DESCRIPTION = "Performance add-ons for the astroML package"
LONG_DESCRIPTION = open('README.rst').read()
NAME = "astroML_addons"
AUTHOR = "Jake VanderPlas"
AUTHOR_EMAIL = "vanderplas@astro.washington.edu"
MAINTAINER = "Jake VanderPlas"
MAINTAINER_EMAIL = "vanderplas@astro.washington.edu"
URL = 'http://astroML.github.com'
DOWNLOAD_URL = 'http://github.com/astroML/astroML'
LICENSE = 'BSD'

# import partial version of the package for version info
import astroML_addons
VERSION = astroML_addons.__version__

def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('astroML_addons')

    return config

setup(configuration=configuration,
      name=NAME,
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
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.6',
        'Topic :: Scientific/Engineering :: Astronomy'],
     )
