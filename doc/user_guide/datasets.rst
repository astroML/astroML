.. _datasets_description:

Data Sets
=========
One of the major components of astroML is its tools for downloading and
working with astronomical data sets.  The available routines are available
in the module :mod:`astroML.datasets`, and details are available in the
documentation of the functions therein.  In this section we will summarize
some of the data sets made available by the code, and show some visualizations
of this data

.. currentmodule:: astroML.datasets

.. _datasets_cache:

Data Set Cache Location
-----------------------
The total size of the data sets is in the hundreds of megabytes, too large to
be bundled with the astroML source distribution.  To make working with
data sets more convenient, astroML contains routines which download the
data from their locations on the web, and cache the results to disk for
future use.

For example, the :func:`fetch_sdss_spectrum` function queries the SDSS
database using an SQL query, and retrieves the data.  The file is stored
to disk in a location that can be defined by the user.  The default location
is ``~/astroML_data``, and this default can be overridden by setting the
``ASTROML_DATA`` environment variable.  Any subsequent time the same function
is called, the cached version of the data is used automatically.

.. _datasets_sdss:

SDSS Data
---------
Much of the data made available by astroML comes from the
`Sloan Digital Sky Survey (SDSS) <http://www.sdss.org>`_, a decade-plus
photometric and spectroscopic survey at the Apache Point Observatory in
New Mexico.  The survey obtained photometry for hundreds of millions of
stars, quasars, and galaxies, and spectra for several million of these
objects.  In addition, the second phase of the survey performed repeated
imaging over a small portion of the sky, called Stripe 82, enabling the
study of the time-variation of many objects.

SDSS photometric data are observed through five filters, `u`, `g`, `r`, `i`,
and `z`.  A visualization of the range of these filters is shown below:


.. image:: ../examples/images/datasets/plot_sdss_filters_1.png
   :target: ../examples/datasets/plot_sdss_filters.html
   :align: center
   :scale: 80


.. _datasets_sdss_spectra:

SDSS Spectra
~~~~~~~~~~~~
The SDSS spectroscopic data is available in a database, indexed by three
numbers: the plate, date, and fiber number.  The :ref:`fetch_sdss_spectrum`
takes a plate, mjd, and fiber, and downloads the spectrum to disk.  The
spectral data can be visualized as follows:

.. image:: ../examples/images/datasets/plot_sdss_spectrum_1.png
   :target: ../examples/datasets/plot_sdss_spectrum.html
   :align: center
   :scale: 80

As with all figures in this documentation, clicking on the image will link
to a page showing the source code used to download the data and plot the
result.

.. _datasets_sdss_photometry:

SDSS Photometry
~~~~~~~~~~~~~~~
Similarly to the spectroscopic data, the photometric data can be accessed
directly using the SQL interface to the Database Archive Server.  astroML
contains a function which accesses this data directly using a python SQL
query tool.  The function is called :func:`fetch_sdss_galaxy_colors` and
can be used as a template for making custom data sets available with a
simple Python command.  Some of the results are visualized below:

.. image:: ../examples/images/datasets/plot_sdss_galaxy_colors_1.png
   :target: ../examples/datasets/plot_sdss_galaxy_colors.html
   :align: center
   :scale: 80


.. _datasets_sdss_corrected_spectra:


SDSS Corrected Spectra
~~~~~~~~~~~~~~~~~~~~~~
The SDSS spectra come from galaxies at a range of redshifts, and have sections
of unreliable or missing data due to sky absorption, cosmic rays, bad detector
pixels, or other effects.  AstroML provides a set of spectra which have been
moved to rest frame, corrected for masking using an iterative PCA reconstruction
technique (see :ref:`example_datasets_compute_sdss_pca`), and resampled to
1000 common wavelength bins.  The spectra can be downloaded using
:func:`fetch_sdss_corrected_spectra`; some examples of these are shown
below:

.. image:: ../examples/images/datasets/plot_corrected_spectra_1.png
   :target: ../examples/datasets/plot_corrected_spectra.html
   :align: center
   :scale: 80

These data are used in several of the example figures from
:ref:`book_fig_chapter7`.

Because these spectra are meant to enable high-dimensional classification and
visualization routines, it is useful to have some extra classification data
for these objects.  One set of features available in the data set is the
line ratio measurements.  These can be visualized as shown below:

.. image:: ../examples/images/datasets/plot_sdss_line_ratios_1.png
   :target: ../examples/datasets/plot_sdss_line_ratios.html
   :align: center
   :scale: 80


.. _datasets_sdss_spec_gals:

SDSS Spectroscopic Sample
~~~~~~~~~~~~~~~~~~~~~~~~~
Along with spectra, SDSS catalogued photometric observations of the objects
in the survey area.  Those objects with both spectra and photometry available
provide a wealth of information about many classes of objects in the night
sky.  The photometry from the SDSS spectroscopic galaxy sample is available
using the routine :func:`fetch_sdss_specgals`, and some of the attributes
are shown in the following visualization:

.. image:: ../examples/images/datasets/plot_sdss_specgals_1.png
   :target: ../examples/datasets/plot_sdss_specgals.html
   :align: center
   :scale: 80

.. _datasets_sdss_imaging:

SDSS Imaging Sample
~~~~~~~~~~~~~~~~~~~
While the spectroscopically observed objects in SDSS offer a large number of
measured features for each object, the total number of observed objects of
each class (star, galaxy, quasar) is under one million.  The full photometric
sample goes much deeper, and thus contains photometric measurements of
hundreds of millions of objects.  astroML has a function called
:func:`fetch_imaging_sample` which loads a selection of this data.  Some
of the returned attributes are visualized below:

.. image:: ../examples/images/datasets/plot_sdss_imaging_1.png
   :target: ../examples/datasets/plot_sdss_imaging.html
   :align: center
   :scale: 80

.. _datasets_sdss_sspp:

SDSS Segue Stellar Parameters Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Several groups have produced various value-added catalogs which contain
additional observed features for objects in the SDSS database.  One example
is the Segue Stellar Parameters Pipeline (SSPP), which makes available a
large number of additional object features derived from SDSS photometry
and spectra.  This data can be downloaded using the function
:func:`fetch_sdss_sspp`.  Some of the metallicity and temperature data is
visualized below:

.. image:: ../examples/images/datasets/plot_SDSS_SSPP_1.png
   :target: ../examples/datasets/plot_SDSS_SSPP.html
   :align: center
   :scale: 100

Many more attributes are available; see the :func:`fetch_sdss_sspp`
documentation for details.

.. _datasets_stripe82_time:

Stripe 82: Time Domain
~~~~~~~~~~~~~~~~~~~~~~
During the second phase of the SDSS, the project repeatedly surveyed a small
swath of sky known as `Stripe 82`.  This yielded an unprecedented set of
data in the time domain, which yielded insight into phenomena as wide-ranging
as the orbits of asteroids, the variability of certain classes of stars,
and the acceleration of the expansion of the universe.

astroML contains two datasets based on Stripe 82 data: one containing
observations of RR-Lyrae stars, and one containing observations of moving
objects (i.e. asteroids) within the solar system.

The RR-Lyrae data can be obtained using the :func:`fetch_rrlyrae_mags`
function, and result in the dataset visualized below:

.. image:: ../examples/images/datasets/plot_rrlyrae_mags_1.png
   :target: ../examples/datasets/plot_rrlyrae_mags.html
   :align: center
   :scale: 80

The moving objects can be obtained using the :func:`fetch_moving_objects`
function, giving a dataset containing not only photometric observations,
but also orbital parameters.  A portion of this information went into the
following visualization:

.. |MOC1| image:: ../examples/images/datasets/plot_moving_objects_1.png
   :target: ../examples/datasets/plot_moving_objects.html
   :scale: 50

.. |MOC2| image:: ../examples/images/datasets/plot_moving_objects_2.png
   :target: ../examples/datasets/plot_moving_objects.html
   :scale: 50

.. centered:: |MOC1| |MOC2|

.. _datasets_stripe82_standard:

Stripe 82: Standard Stars
~~~~~~~~~~~~~~~~~~~~~~~~~
Along with time-domain data, the repeated observations in Stripe 82 enabled
stacked photometry of sources to minimize the statistical error in their
measured fluxes.  The Stripe 82 standard stars are a set of stars in this
region which are below a specified variability criterion.  The multiple
exposures were combined to yield a highly precise catalog of stars.  THis
data can be obtained using the :func:`fetch_sdss_S82standards` function.
Some of the data in this catalog is visualized below:

.. image:: ../examples/images/datasets/plot_sdss_S82standards_1.png
   :target: ../examples/datasets/plot_sdss_S82standards.html
   :align: center
   :scale: 80
