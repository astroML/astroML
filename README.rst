.. -*- mode: rst -*-

=======================================
AstroML: Machine Learning for Astronomy
=======================================

.. image:: http://img.shields.io/badge/arXiv-1411.5039-orange.svg?style=flat
        :target: http://arxiv.org/abs/1411.5039
.. image:: http://img.shields.io/travis/astroML/astroML/master.svg?style=flat
        :target: https://travis-ci.org/astroML/astroML/
.. image:: http://img.shields.io/pypi/v/astroML.svg?style=flat
        :target: https://pypi.python.org/pypi/astroML
.. image:: http://img.shields.io/pypi/dm/astroML.svg?style=flat
        :target: https://pypi.python.org/pypi/astroML
.. image:: http://img.shields.io/badge/license-BSD-blue.svg?style=flat
        :target: https://github.com/astroml/astroml/blob/master/LICENSE

AstroML is a Python module for machine learning and data mining
built on numpy, scipy, scikit-learn, and matplotlib,
and distributed under the 3-Clause BSD license.
It contains a growing library of statistical and machine learning
routines for analyzing astronomical data in python, loaders for several open
astronomical datasets, and a large suite of examples of analyzing and
visualizing astronomical datasets.

This project was started in 2012 by Jake VanderPlas to accompany the book
*Statistics, Data Mining, and Machine Learning in Astronomy* by
Zeljko Ivezic, Andrew Connolly, Jacob VanderPlas, and Alex Gray.

Core and Addons
===============
The project is split into two components.  The core ``astroML`` library is
written in python only, and is designed to be very easy to install for
any users, even those who don't have a working C or fortran compiler.
A companion library, ``astroML_addons``, can be optionally installed for
increased performance on certain algorithms.  Every algorithm
in ``astroML_addons`` has a pure python counterpart in the
core ``astroML`` implementation, but the ``astroML_addons`` library
contains faster and more efficient implementations in compiled code.
Furthermore, if ``astroML_addons`` is installed on your system, the core
``astroML`` library will import and use the faster routines by default.

The reason for this split is the ease of use for newcomers to Python.  If the
prerequisites are already installed on your system, the core ``astroML``
library can be installed and used on any system with little trouble.  The
``astroML_addons`` library requires a C compiler, but is also designed to be
easy to install for more advanced users.  See further discussion in
"Development", below.


Important Links
===============
- HTML documentation: http://www.astroML.org
- Core source-code repository: http://github.com/astroML/astroML
- Addons source-code repository: http://github.com/astroML/astroML_addons
- Issue Tracker: http://github.com/astroML/astroML/issues
- Mailing List: https://groups.google.com/forum/#!forum/astroml-general


Installation
============

This package uses distutils, which is the default way of installing python
modules.  **Before installation, make sure your system meets the prerequisites
listed in Dependencies, listed below.**

Core
----
To install the core ``astroML`` package in your home directory, use::

  pip install astroML

The core package is pure python, so installation should be straightforward
on most systems.  To install from source, use::

  python setup.py install

You can specify an arbitrary directory for installation using::

  python setup.py install --prefix='/some/path'

To install system-wide on Linux/Unix systems::

  python setup.py build
  sudo python setup.py install

Addons
------
The ``astroML_addons`` package requires a working C/C++ compiler for
installation.  It can be installed using::

  pip install astroML_addons

To install from source, refer to http://github.com/astroML/astroML_addons


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

- Python_ version 2.6-2.7 and 3.3+
- Numpy_ >= 1.4
- Scipy_ >= 0.7
- Scikit-learn_ >= 0.10
- Matplotlib_ >= 0.99
- AstroPy_ > 0.2.5
  AstroPy is required to read Flexible Image Transport
  System (FITS) files, which are used by several datasets.
  
This configuration matches the Ubuntu 10.04 LTS release from April 2010,
with the addition of scikit-learn.

To run unit tests, you will also need nose >= 0.10

Add-on Dependencies
-------------------
The fast code in ``astroML_addons`` requires a working C/C++ compiler.

Optional Dependencies
---------------------
Several of the example scripts require specialized or upgraded packages.
These requirements are listed at the top of the particular scripts

- Scipy_ version 0.11 added a sparse graph submodule.
  The minimum spanning tree example requires scipy >= 0.11

- PyMC_ provides a nice interface for Markov-Chain Monte Carlo. Several astroML
  examples use pyMC for exploration of high-dimensional spaces. The examples
  were written with pymc version 2.2

- HEALPy_ provides an interface to
  the HEALPix pixelization scheme, as well as fast spherical harmonic
  transforms.

Development
===========
This package is designed to be a repository for well-written astronomy code,
and submissions of new routines are encouraged.  After installing the
version-control system Git_, you can check out
the latest sources from GitHub_ using::

  git clone git://github.com/astroML/astroML.git

or if you have write privileges::

  git clone git@github.com:astroML/astroML.git

Contribution
------------
We strongly encourage contributions of useful astronomy-related code:
for `astroML` to be a relevant tool for the python/astronomy community,
it will need to grow with the field of research.  There are a few
guidelines for contribution:

General
~~~~~~~
Any contribution should be done through the github pull request system (for
more information, see the
`help page <https://help.github.com/articles/using-pull-requests>`_
Code submitted to ``astroML`` should conform to a BSD-style license,
and follow the `PEP8 style guide <http://www.python.org/dev/peps/pep-0008/>`_.

Documentation and Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~
All submitted code should be documented following the
`Numpy Documentation Guide`_.  This is a unified documentation style used
by many packages in the scipy universe.

In addition, it is highly recommended to create example scripts that show the
usefulness of the method on an astronomical dataset (preferably making use
of the loaders in ``astroML.datasets``).  These example scripts are in the
``examples`` subdirectory of the main source repository.

Add-on code
~~~~~~~~~~~
We made the decision early-on to separate the core routines from
high-performance compiled routines.
This is to make sure that installation of the core
package is as straightforward as possible (i.e. not requiring a C compiler).

Contributions of efficient compiled code to ``astroML_addons`` is encouraged:
the availability of efficient implementations of common algorithms in python
is one of the strongest features of the python universe.  The preferred
method of wrapping compiled libraries is to use
`cython <http://www.cython.org>`_; other options (weave, SWIG, etc.) are
harder to build and maintain.

Currently, the policy is that any efficient algorithm included in
``astroML_addons`` should have a duplicate python-only implementation in
``astroML``, with code that selects the faster routine if it's available.
(For an example of how this works, see the definition of the ``lomb_scargle``
function in ``astroML/periodogram.py``).
This policy exists for a few reasons:

1. it allows novice users to have all the functionality of ``astroML`` without
   requiring the headache of complicated installation steps.
2. it serves a didactic purpose: python-only implementations are often easier
   to read and understand than equivalent implementations in C or cython.
3. it enforces the good coding practice of avoiding premature optimization.
   First make sure the code works (i.e. write it in simple python).  Then
   create an optimized version in the addons.

If this policy proves especially burdensome in the future, it may be revisited.

.. _Numpy Documentation Guide: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

Authors
=======

Package Author
--------------
* Jake Vanderplas <vanderplas@astro.washington.edu> http://jakevdp.github.com

Code Contribution
-----------------
* Morgan Fouesneau https://github.com/mfouesneau
* Julian Taylor http://github.com/juliantaylor


.. _Python: http://www.python.org
.. _Numpy: http://www.numpy.org
.. _Scipy: http://www.scipy.org
.. _Scikit-learn: http://scikit-learn.org
.. _Matplotlib: http://matplotlib.org
.. _AstroPy: http://www.astropy.org/
.. _PyMC: http://pymc-devs.github.com/pymc/
.. _HEALPy: https://github.com/healpy/healpy>
.. _Git: http://git-scm.com/
.. _GitHub: http://www.github.com
