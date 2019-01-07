0.4rc1 (2019-01-07)
===================

API Changes and Other Changes
-----------------------------

- Removed deprecated KDE class. [#119]

- Switched to use the updated scikit-learn API for GaussianMixture. This
  change depends on scikit-learn 0.18+. [#125]

- Minimum required astropy version is now 1.1. [#142]

- The book and paper figures has been moved out to their separate
  repository (``astroML_figures``).


0.3
===
- Add support for Python 3
- Add continuous integration via Travis
- Bug: correctly account for errors in Ridge/Lasso regression
- Add figure tests in ``compare_images.py``

0.2
===
- Documentation and example updates
- Moved from using ``pyfits`` to using ``astropy.io.fits``
- Fix the prior for the Bayesian Blocks algorithm

0.1.1
=====
*Bug fixes, January 2013*
- Fixed errors in dataset downloaders: they failed on some platforms
- Added citation information to the website
- Updated figures to reflect those submitted for publication
- Performance improvement in ``freedman_bin_width``
- Fix setup issue when sklearn is not installed
- Enhancements to ``devectorize_axes`` function

0.1
===
*Initial release, October 2012*
