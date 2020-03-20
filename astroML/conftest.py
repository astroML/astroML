try:
    from pytest_astropy_header.display import (PYTEST_HEADER_MODULES,
                                               TESTED_VERSIONS)

    def pytest_configure(config):
        config.option.astropy_header = True
    try:
        PYTEST_HEADER_MODULES['Cython'] = 'Cython'
        PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
        PYTEST_HEADER_MODULES['scikit-learn'] = 'sklearn'
        PYTEST_HEADER_MODULES['pymc3'] = 'pymc3'
        PYTEST_HEADER_MODULES['astroML_addons'] = 'astroML_addons'
        del PYTEST_HEADER_MODULES['h5py']
        del PYTEST_HEADER_MODULES['Pandas']
    except (KeyError):
        pass

    # This is to figure out the package version, rather than
    # using Astropy's

    from . import __version__ as version

    TESTED_VERSIONS['astroMl'] = version

except ImportError:
    pass
