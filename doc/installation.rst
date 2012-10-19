=======================
Installation of astroML
=======================



Important Links
===============
- Source-code repository: http://github.com/astroML/astroML
- HTML documentation: http://astroML.github.com


Dependencies
============
There are three levels of dependencies in astroML.  *Core* dependencies are
required for the core ``astroML`` package.  *Add-on* dependencies are required
for the performance ``astroML_addons``.  *Optional* dependencies are required
to run some (but not all) of the example scripts.  Individual example scripts
will list their optional dependencies at the top of the file.

Core Dependencies
-----------------
The core ``astroML`` package requires the following:

- `Python <http://python.org>`_ version 2.6.x - 2.7.x
  (astroML does not yet support python 3.x)
- `Numpy <http://numpy.scipy.org/>`_ >= 1.4
- `Scipy <http://www.scipy.org/>`_ >= 0.7
- `matplotlib <http://matplotlib.org/>`_ >= 0.99
- `pyfits <http://www.stsci.edu/institute/software_hardware/pyfits>`_ >= 3.0.
  PyFITS is a python reader for Flexible Image Transport
  System (FITS) files, based on cfitsio.  Several of the dataset loaders
  require pyfits.

This configuration matches the Ubuntu 10.04 LTS release from April 2010.

To run unit tests, you will also need nose >= 0.10

Add-on Dependencies
-------------------
The fast code in ``astroML_performance`` requires a working C/C++ compiler.

Optional Dependencies
---------------------
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

Installation
============

This package uses distutils, which is the default way of installing python
modules.

Core
----
To install the core ``astroML`` package in your home directory, use::

  python setup.py install --home

You can specify an arbitrary directory for installation using::

  python setup.py install --prefix='/some/path'

To install system-wide on Linux/Unix systems::

  python setup.py build
  sudo python setup.py install

Addons
------
The ``astroML_addons`` package requires a working C/C++ compiler for
installation.  It can be installed using::

  python setup_addons.py install

The script can make use of any of the extra options discussed above.
