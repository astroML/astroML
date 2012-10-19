.. astro-ml documentation master file, created by
   sphinx-quickstart on Thu Oct  6 15:37:12 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AstroML: Machine Learning code for Astronomy
============================================

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

The project is split into two components.  The core ``astroML`` library is
written in python only, and is designed to be very easy to install for
any users, even those who don't have a working C or fortran compiler.
A companion library, ``astroML_addons``, can be optionally installed for
increased performance on certain algorithms.  Every algorithm
in ``astroML_addons`` exists in the core ``astroML`` implementation, but the
``astroML_addons`` library contains faster and more efficient implementations.
Furthermore, if ``astroML_addons`` is installed on your system, the core
``astroML`` library will import and use the faster routines by default.

Example Plot Gallery
====================

.. toctree::
   :maxdepth: 2
   :numbered:

   auto_examples/index


Figures from Text
=================

.. toctree::
   :maxdepth: 2

   auto_book_figures/index

Figures from Papers using AstroML
=================================

.. toctree::
   :maxdepth: 2

   auto_paper_figures/index

Information
===========

.. toctree::
   :maxdepth: 2

   installation
   development

Appendix
========

.. toctree::
   :maxdepth: 2

   contents

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Authors
=======
Jake Vanderplas <vanderplas@astro.washington.edu> http://jakevdp.github.com
