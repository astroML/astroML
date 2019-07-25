Obtaining the Source Code
=========================
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


.. _Numpy Documentation Guide: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
