1.0a1 (2020-03-23)
==================

- Added LinearRegressionwithErrors to handle errors in both dependent and
  independent variables using pymc3. [#206]

- Removed suppport for Python versions <3.5. [#174]

- Deprecated function ``savitzky_golay`` in favour of the scipy
  implementation. [#193]

- Deprecated functions ``check_random_state`` and ``logsumexp`` in favour of
  their equivalent in scikit-learn and scipy, respectively. [#190]


0.4.1 (2019-10-01)
==================

- Fix syntax for matplotlib.rc usage. [#188]

- Various code cleanups and updates to the website.


0.4 (2019-03-06)
================

- New utils subpackage, including deprecated decorator and new warning
  types. ``astroML.decorators`` has been moved to this subpackage. [#141]


API Changes and Other Changes
-----------------------------

- Removed deprecated KDE class. [#119]

- Switched to use the updated scikit-learn API for GaussianMixture. This
  change depends on scikit-learn 0.18+. [#125]

- Minimum required astropy version is now 1.2. [#173]

- Deprecated ``astroML.cosmology.Cosmology()`` in favour of
  ``astropy.cosmology``. [#121]

- Deprecated the Lomb-Scargle periodograms from ``astroML.time_series`` in
  favour of ``astropy.stats.LombScargle``. [#173]

- Deprecated histograms, including Bayesian blocks, as they have been moved
  to ``astropy.stats``. [#142]

- The book and paper figures has been moved out to their separate
  repository (``astroML_figures``).


0.3 (2015-01-28)
================

- Add support for Python 3
- Add continuous integration via Travis
- Bug: correctly account for errors in Ridge/Lasso regression
- Add figure tests in ``compare_images.py``

0.2 (2013-12-09)
================

- Documentation and example updates
- Moved from using ``pyfits`` to using ``astropy.io.fits``
- Fix the prior for the Bayesian Blocks algorithm

0.1.1 (2013-01-19)
==================

Bug fixes, January 2013
-----------------------

- Fixed errors in dataset downloaders: they failed on some platforms
- Added citation information to the website
- Updated figures to reflect those submitted for publication
- Performance improvement in ``freedman_bin_width``
- Fix setup issue when sklearn is not installed
- Enhancements to ``devectorize_axes`` function

0.1 (2012-10)
=============

Initial release, October 2012
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
