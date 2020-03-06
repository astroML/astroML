.. -*- mode: rst -*-

=======================================
AstroML: Machine Learning for Astronomy
=======================================

.. image:: https://img.shields.io/badge/arXiv-1411.5039-orange.svg?style=flat
        :target: https://arxiv.org/abs/1411.5039
.. image:: https://img.shields.io/travis/astroML/astroML/master.svg?style=flat
        :target: https://travis-ci.org/astroML/astroML/
.. image:: https://img.shields.io/pypi/v/astroML.svg?style=flat
        :target: https://pypi.python.org/pypi/astroML
.. image:: https://img.shields.io/pypi/dm/astroML.svg?style=flat
        :target: https://pypi.python.org/pypi/astroML
.. image:: https://img.shields.io/badge/license-BSD-blue.svg?style=flat
        :target: https://github.com/astroml/astroml/blob/master/LICENSE

AstroML is a Python module for machine learning and data mining
built on numpy, scipy, scikit-learn, and matplotlib,
and distributed under the BSD license.
It contains a growing library of statistical and machine learning
routines for analyzing astronomical data in python, loaders for several open
astronomical datasets, and a large suite of examples of analyzing and
visualizing astronomical datasets.

This project was started in 2012 by Jake VanderPlas to accompany the book
*Statistics, Data Mining, and Machine Learning in Astronomy* by
Zeljko Ivezic, Andrew Connolly, Jacob VanderPlas, and Alex Gray.


Important Links
===============
- HTML documentation: https://www.astroML.org
- Core source-code repository: https://github.com/astroML/astroML
- Figure source-code repository: https://github.com/astroML/astroML-figures
- Issue Tracker: https://github.com/astroML/astroML/issues
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

A conda package for astroML is also available either on the conda-forge or
on the astropy conda channels::

  conda install -c astropy astroML

The core package is pure python, so installation should be straightforward
on most systems.  To install from source, use::

  python setup.py install

You can specify an arbitrary directory for installation using::

  python setup.py install --prefix='/some/path'

To install system-wide on Linux/Unix systems::

  python setup.py build
  sudo python setup.py install


Dependencies
============
There are two levels of dependencies in astroML.  *Core* dependencies are
required for the core ``astroML`` package. *Optional* dependencies are required
to run some (but not all) of the example scripts.  Individual example scripts
will list their optional dependencies at the top of the file.

Core Dependencies
-----------------
The core ``astroML`` package requires the following (some of the
functionality might work with older versions):

- Python_ version 3.5+
- Numpy_ >= 1.8
- Scipy_ >= 0.11
- Scikit-learn_ >= 0.18
- Matplotlib_ >= 2.1.1
- AstroPy_ >= 1.2


Optional Dependencies
---------------------
Several of the example scripts require specialized or upgraded packages.
These requirements are listed at the top of the particular scripts

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

.. _Numpy Documentation Guide: https://numpydoc.readthedocs.io/en/latest/format.html

Authors
=======

Package Author
--------------
* Jake Vanderplas https://github.com/jakevdp
  http://jakevdp.github.com

Maintainer
----------
* Brigitta Sipocz https://github.com/bsipocz

Code Contribution
-----------------
* Morgan Fouesneau https://github.com/mfouesneau
* Julian Taylor http://github.com/juliantaylor


.. _Python: https://www.python.org
.. _Numpy: https://www.numpy.org
.. _Scipy: https://www.scipy.org
.. _Scikit-learn: https://scikit-learn.org
.. _Matplotlib: https://matplotlib.org
.. _AstroPy: http://www.astropy.org/
.. _HEALPy: https://github.com/healpy/healpy
.. _Git: https://git-scm.com/
.. _GitHub: https://www.github.com
