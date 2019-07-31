===============
Class reference
===============

This is a list of modules, classes, and functions available in ``astroML``.
For more details, please refer to the :ref:`user guide <user_guide>` or
the :ref:`text book <textbook>`.  Examples of the use of ``astroML``
can also be found in the :ref:`code examples <example_root>`, the
:ref:`text book figures <book_fig_root>` and the
:ref:`paper figures <paper_fig_root>`.

.. automodule:: astroML

Plotting Functions: :mod:`astroML.plotting`
===========================================

.. automodule:: astroML.plotting
   :no-members:
   :no-inherited-members:

Functions
---------
.. currentmodule:: astroML

.. autosummary::
   :toctree: generated/
   :template: function.rst

   plotting.hist
   plotting.scatter_contour

Classes
-------
.. autosummary::
   :toctree: generated/
   :template: class.rst

   plotting.MultiAxes

Density Estimation & Histograms: :mod:`astroML.density_estimation`
==================================================================

.. automodule:: astroML.density_estimation
   :no-members:
   :no-inherited-members:

Histogram Tools
---------------
.. currentmodule:: astroML

.. autosummary::
   :toctree: generated/
   :template: function.rst

   density_estimation.histogram
   density_estimation.bayesian_blocks
   density_estimation.knuth_bin_width
   density_estimation.scotts_bin_width
   density_estimation.freedman_bin_width


Density Estimation
------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   density_estimation.XDGMM
   density_estimation.KNeighborsDensity
   density_estimation.EmpiricalDistribution
   density_estimation.FunctionDistribution


Linear Regression & Fitting: :mod:`astroML.linear_model`
========================================================

.. automodule:: astroML.linear_model
   :no-members:
   :no-inherited-members:

Linear Regression
-----------------
.. currentmodule:: astroML

.. autosummary::
   :toctree: generated/
   :template: class.rst

   linear_model.LinearRegression
   linear_model.PolynomialRegression
   linear_model.BasisFunctionRegression
   linear_model.NadarayaWatson

Functions
---------
.. currentmodule:: astroML

.. autosummary::
   :toctree: generated/
   :template: function.rst

   linear_model.TLS_logL


Loading of Datasets: :mod:`astroML.datasets`
============================================

.. automodule:: astroML.datasets
   :no-members:
   :no-inherited-members:

.. currentmodule:: astroML

.. autosummary::
   :toctree: generated/
   :template: function.rst

   datasets.fetch_sdss_spectrum
   datasets.fetch_sdss_corrected_spectra
   datasets.fetch_sdss_S82standards
   datasets.fetch_dr7_quasar
   datasets.fetch_moving_objects
   datasets.fetch_sdss_galaxy_colors
   datasets.fetch_nasa_atlas
   datasets.fetch_sdss_sspp
   datasets.fetch_sdss_specgals
   datasets.fetch_great_wall
   datasets.fetch_imaging_sample
   datasets.fetch_wmap_temperatures
   datasets.fetch_rrlyrae_mags
   datasets.fetch_rrlyrae_combined
   datasets.fetch_LINEAR_sample
   datasets.fetch_LINEAR_geneva
   datasets.fetch_LIGO_bigdog
   datasets.fetch_LIGO_large
   datasets.fetch_hogg2010test
   datasets.fetch_rrlyrae_templates
   datasets.fetch_sdss_filter
   datasets.fetch_vega_spectrum
   datasets.generate_mu_z


Time Series Analysis: :mod:`astroML.time_series`
================================================

.. automodule:: astroML.time_series
   :no-members:
   :no-inherited-members:


Periodic Time Series
--------------------

.. currentmodule:: astroML

.. autosummary::
   :toctree: generated/
   :template: function.rst

   time_series.lomb_scargle
   time_series.lomb_scargle_bootstrap
   time_series.multiterm_periodogram
   time_series.search_frequencies

.. currentmodule:: astroML

.. autosummary::
   :toctree: generated/
   :template: class.rst

   time_series.MultiTermFit

Aperiodic Time Series
---------------------

.. currentmodule:: astroML

.. autosummary::
   :toctree: generated/
   :template: function.rst

   time_series.ACF_scargle
   time_series.ACF_EK
   time_series.generate_power_law
   time_series.generate_damped_RW


Statistical Functions: :mod:`astroML.stats`
===========================================

.. automodule:: astroML.stats
   :no-members:
   :no-inherited-members:

.. currentmodule:: astroML

.. autosummary::
   :toctree: generated/
   :template: function.rst

   stats.binned_statistic
   stats.binned_statistic_2d
   stats.binned_statistic_dd
   stats.sigmaG
   stats.median_sigmaG
   stats.mean_sigma
   stats.fit_bivariate_normal
   stats.bivariate_normal
   stats.trunc_exp
   stats.linear


Dimensionality Reduction: :mod:`astroML.dimensionality`
=======================================================

.. automodule:: astroML.dimensionality
   :no-members:
   :no-inherited-members:

.. currentmodule:: astroML

.. autosummary::
   :toctree: generated/
   :template: function.rst

   dimensionality.iterative_pca


Correlation Functions: :mod:`astroML.correlation`
=================================================

.. automodule:: astroML.correlation
   :no-members:
   :no-inherited-members:

.. currentmodule:: astroML

.. autosummary::
   :toctree: generated/
   :template: function.rst

   correlation.two_point
   correlation.two_point_angular
   correlation.bootstrap_two_point
   correlation.bootstrap_two_point_angular


Filters: :mod:`astroML.filters`
===============================

.. automodule:: astroML.filters
   :no-members:
   :no-inherited-members:

.. currentmodule:: astroML

.. autosummary::
   :toctree: generated/
   :template: function.rst

   filters.savitzky_golay
   filters.wiener_filter
   filters.min_component_filter


Fourier and Wavelet Transforms: :mod:`astroML.fourier`
======================================================

.. automodule:: astroML.fourier
   :no-members:
   :no-inherited-members:

.. currentmodule:: astroML

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fourier.FT_continuous
   fourier.IFT_continuous
   fourier.PSD_continuous
   fourier.wavelet_PSD
   fourier.sinegauss
   fourier.sinegauss_FT
   fourier.sinegauss_PSD


Luminosity Functions: :mod:`astroML.lumfunc`
============================================

.. automodule:: astroML.lumfunc
   :no-members:
   :no-inherited-members:

.. currentmodule:: astroML

.. autosummary::
   :toctree: generated/
   :template: function.rst

   lumfunc.Cminus
   lumfunc.binned_Cminus
   lumfunc.bootstrap_Cminus

Classification: :mod:`astroML.classification`
=============================================

.. automodule:: astroML.classification
   :no-members:
   :no-inherited-members:

.. currentmodule:: astroML

.. autosummary::
   :toctree: generated/
   :template: class.rst

   classification.GMMBayes

Resampling: :mod:`astroML.resample`
===================================

.. automodule:: astroML.resample
   :no-members:
   :no-inherited-members:

.. currentmodule:: astroML

.. autosummary::
   :toctree: generated/
   :template: function.rst

   resample.bootstrap
   resample.jackknife
