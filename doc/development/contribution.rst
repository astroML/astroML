Bleeding-edge Source
====================
This package is designed to be a repository for well-written astronomy code,
and submissions of new routines are encouraged.  After installing the
version-control system `git <http://git-scm.com/>`_, you can check out
the latest sources from `github <http://github.com>`_ using::

  git clone git://github.com/astroML/astroML.git

or if you have write privileges::

  git clone git@github.com:astroML/astroML.git

Contribution
============
We strongly encourage contributions of useful astronomy-related code:
for `astroML` to be a relevant tool for the python/astronomy community,
it will need to grow with the field of research.  There are a few
guidelines for contribution:

General
-------
Any contribution should be done through the github pull request system (for
more information, see the
`help page <https://help.github.com/articles/using-pull-requests>`_
Code submitted to ``astroML`` should conform to a BSD-style license,
and follow the `PEP8 style guide <http://www.python.org/dev/peps/pep-0008/>`_.

Tests
-----
All submitted code should be tested using the nose testing framework.  For
examples of how these tests work, see the ``tests`` within the ``astroML``
package and each of its submodules.

Documentation and Examples
--------------------------
All submitted code should be documented following the
`Numpy Documentation Guide`_.  This is a unified documentation style used
by many packages in the scipy universe.

In addition, it is highly recommended to create example scripts that show the
usefulness of the method on an astronomical dataset (preferably making use
of datasets available through ``astroML.datasets``).
Some of these example scripts can be seen in the ``examples`` subdirectory
of the main source repository: :ref:`examples_root`.

Add-on code
-----------
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
This policy exists for two reasons:

 1. it allows novice users to have all the functionality of ``astroML`` without
    requiring the headache of complicated installation steps.
 2. it serves a didactic purpose: python-only implementations are often easier
    to read and understand than equivalent implementations in C or cython.

If this policy proves especially burdensome in the future, it may be revisited.

.. _Numpy Documentation Guide: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
