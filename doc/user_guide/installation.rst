.. _astroML_installation:

Installation of astroML
=======================

The astroML project is split into two components.  The core ``astroML``
library is
written in python only, and is designed to be very easy to install for
any users, even those who don't have a working C or fortran compiler.
A companion library, ``astroML_addons``, can be optionally installed for
increased performance on certain algorithms.  Every algorithm
in ``astroML_addons`` exists in the core ``astroML`` implementation, but the
``astroML_addons`` library contains faster and more efficient implementations.
Furthermore, if ``astroML_addons`` is installed on your system, the core
``astroML`` library will import and use the faster routines by default.


Important Links
---------------
- Source-code repository: http://github.com/astroML/astroML
- HTML documentation: http://astroML.github.com


Installation
------------

This package uses distutils, which is the default way of installing python
modules.  The first step is to obtain the source code: we recommend
downloading it from the
`github repository <http://github.com/astroML/astroML>`_ shown above.
Additionally, you must make sure the :ref:`dependencies <astroML_dependencies>`
are filled: lacking some
of these dependencies will not affect installation, but will affect the
ability to execute the code and examples

Both the astroML and astroML_addons packages are installed using python's
distutils.  The generic commands for installation are as follows::

  python setup.py build
  python setup.py install

The first line builds the source (i.e. for astroML_addons, this compiles the
C code used in the package).  The second line
installs the package so that you can use it from Python.  The default
install location is in your ``site_packages`` or ``dist_packages`` directory
in your default python path.

If you are on a machine without write access to the default installation
location, the location can be specified when installing.  For example, to
install the package in your home directory, use::

  python setup.py install --home

You can specify an arbitrary directory for installation using::

  python setup.py install --prefix='/some/path'


Core
~~~~
The core astroML package is pure python, so the build phase is trivial.  The
installation script is ``setup.py``, and the code can be installed as follows::

    python setup.py build
    python setup.py install


Addons
~~~~~~
The ``astroML_addons`` package requires a working C/C++ compiler for
installation.  It can be installed using::

    python setup_addons.py build
    python setup_addons.py install

Testing
~~~~~~~
After installation, unit tests can be run using the
`nose <https://nose.readthedocs.org/en/latest/>`_ testing framework, either
by typing ``nosetests astroML``, or by typing ``make test`` in the source
directory.  The latter will also run doc tests on the user guide.

.. _astroML_dependencies:

Dependencies
------------
There are three levels of dependencies in astroML.  *Core* dependencies are
required for the core ``astroML`` package.  *Add-on* dependencies are required
for the performance ``astroML_addons``.  *Optional* dependencies are required
to run some (but not all) of the example scripts.  Individual example scripts
will list their optional dependencies at the top of the file.

Core Dependencies
~~~~~~~~~~~~~~~~~
The core ``astroML`` package requires the following:

- `Python <http://python.org>`_ version 2.6.x - 2.7.x
  (astroML does not yet support python 3.x)
- `Numpy <http://numpy.scipy.org/>`_ >= 1.4
- `Scipy <http://www.scipy.org/>`_ >= 0.7
- `scikit-learn <http://scikit-learn.org/>`_ >= 0.10
- `matplotlib <http://matplotlib.org/>`_ >= 0.99
- `pyfits <http://www.stsci.edu/institute/software_hardware/pyfits>`_ >= 3.0.
  PyFITS is a python reader for Flexible Image Transport
  System (FITS) files, based on cfitsio.  Several of the dataset loaders
  require pyfits.

This configuration matches the Ubuntu 10.04 LTS release from April 2010,
with the addition of scikit-learn.

To run unit tests, you will also need nose >= 0.10

Add-on Dependencies
~~~~~~~~~~~~~~~~~~~
The fast code in ``astroML_performance`` requires a working C/C++ compiler.

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~
Several of the example scripts require specialized or upgraded packages.  These
requirements are listed at the top of the example scripts.

- `scipy version 0.11 <http://www.scipy.org>`_ added a sparse graph submodule.
  The minimum spanning tree example requires scipy >= 0.11
- `pyMC <http://pymc-devs.github.com/pymc/>`_
  provides a nice interface for Markov-Chain Monte Carlo.  Several examples
  use pyMC for exploration of high-dimensional spaces.  The examples
  were written with pymc version 2.2
- `healpy <https://github.com/healpy/healpy>`_ provides an interface to
  the HEALPix pixelization scheme, as well as fast spherical harmonic
  transforms.
