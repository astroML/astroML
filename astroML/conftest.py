
try:
    from astropy.tests.plugins.display import (pytest_report_header,
                                               PYTEST_HEADER_MODULES,
                                               TESTED_VERSIONS)
except ImportError:
    # When using astropy 2.0
    from astropy.tests.pytest_plugins import (pytest_report_header,
                                              PYTEST_HEADER_MODULES,
                                              TESTED_VERSIONS)

try:
    PYTEST_HEADER_MODULES['Cython'] = 'Cython'
    PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
    PYTEST_HEADER_MODULES['scikit-learn'] = 'sklearn'
    # pymc import here triggers pytest INTERNALERROR thus as a temporary
    # measure we don't list its version number
    #    PYTEST_HEADER_MODULES['pymc'] = 'pymc'
    PYTEST_HEADER_MODULES['astroML_addons'] = 'astroML_addons'
    del PYTEST_HEADER_MODULES['h5py']
    del PYTEST_HEADER_MODULES['Pandas']
except (KeyError):
    pass


# This is to figure out the package version, rather than
# using Astropy's

from . import __version__ as version

TESTED_VERSIONS['astroMl'] = version
