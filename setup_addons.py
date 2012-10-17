import os
import numpy
from numpy.distutils.core import setup

# import partial version of the package
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
      name='astroML_addons',
      version=VERSION,
      description='Add-ons to astroML package',
      author='Jake VanderPlas',
      author_email='vanderplas@astro.washington.edu',
      url='http://astroML.github.com',
      license='bsd',
     )
