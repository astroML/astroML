.. _astroML_installation:

Installation of astroML
=======================

The astroML project is split into two components.  The core ``astroML``
library is
written in python only, and is designed to be very easy to install for
any users, even those who don't have a working C or fortran compiler.


Important Links
---------------
- Source-code repository: http://github.com/astroML/astroML
- Source-code for book figures: http://github.com/astroML/astroML-figures/
- HTML documentation: http://astroML.github.com
- Python Package Index: http://pypi.python.org/pypi/astroML/


Installation
------------

Python Package Index
~~~~~~~~~~~~~~~~~~~~
The easiest way to install astroML is to use the Python Package Index ``pip``
command.  First make sure the :ref:`dependencies <astroML_dependencies>`
are fulfilled: lacking some of these may not affect installation, but it
will affect the ability to execute code and examples.  Next, use the ``pip``
command to install the packages::

  pip install astroML

(For information about ``pip``, see http://pypi.python.org/pypi/pip)
The first package is python-only, and should install easily on any system.

Conda
~~~~~
AstroML is also available as a conda package via the conda-forge or astropy
channels.
To install with conda, use the following command::

  conda install -c astropy astroML


From Source
~~~~~~~~~~~
To install the latest version from source, we recommend downloading astroML
from the
`github repository <http://github.com/astroML/astroML>`_ shown above.
You must first make sure the :ref:`dependencies <astroML_dependencies>`
are filled: lacking some
of these dependencies will not affect installation, but will affect the
ability to execute the code and examples

The astroML package is installed using python's
distutils.  The generic commands for installation are as follows::

  python setup.py build
  python setup.py install

The default install location is in your ``site_packages`` or
``dist_packages`` directory in your default python path.

If you are on a machine without write access to the default installation
location, the location can be specified when installing.  For example,
you can specify an arbitrary directory for installation using::

  python setup.py install --prefix='/some/path'


Testing
~~~~~~~
After installation, unit tests can be run using the `pytest
<https://pytest.org>`_ testing framework, by typing ``pytest astroML``

.. _astroML_dependencies:

Dependencies
------------
There are two levels of dependencies in astroML.  *Core* dependencies are
required for the core ``astroML`` package.  *Optional* dependencies are
required to run some (but not all) of the example scripts.  Individual
example scripts will list their optional dependencies at the top of the
file.

Core Dependencies
~~~~~~~~~~~~~~~~~
The core ``astroML`` package requires the following:

- `Python <http://python.org>`_ version 3.5+
- `Numpy <http://numpy.scipy.org/>`_ >= 1.13
- `Scipy <http://www.scipy.org/>`_ >= 0.18
- `scikit-learn <http://scikit-learn.org/>`_ >= 0.18
- `matplotlib <http://matplotlib.org/>`_ >= 3.0
- `astropy <http://www.astropy.org/>`_ >= 3.0

To run unit tests, you will also need pytest.

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~
Several of the example scripts require specialized or upgraded packages.  These
requirements are listed at the top of the example scripts.

- `pyMC3 https://docs.pymc.io/`_
  provides a nice interface for Markov-Chain Monte Carlo.  Several examples
  use pyMC3 for exploration of high-dimensional spaces.

- `healpy <https://github.com/healpy/healpy>`_ provides an interface to
  the HEALPix pixelization scheme, as well as fast spherical harmonic
  transforms.
